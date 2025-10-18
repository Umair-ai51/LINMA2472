module VectReverse

# Export key public functions
export gradient, hvp, hessian

# Import dependencies
import ..Flatten                       # Utility to flatten inputs for differentiation
import Base: tanh                      # Extend Base.tanh
import ..Forward: Dual                 # Dual numbers from forward mode AD
import ..Forward: hessian as fwd_hessian  # Forward-mode Hessian implementation


# ==========================================================
# ===  COMPUTATIONAL GRAPH NODE STRUCTURE (VectNode)     ===
# ==========================================================

# Each VectNode represents one node in a computation graph.
# It holds the operation, its arguments, the computed value,
# and a place to store the derivative (for backpropagation).
mutable struct VectNode
    op::Union{Nothing,Symbol}   # Operation type (:+, :*, :tanh, etc.)
    args::Vector{Any}           # Arguments (can be VectNode or constants)
    value::Any                  # Computed value of this node
    derivative::Any             # Gradient/derivative accumulator
end

# Constructor for nodes with an operation and value
function VectNode(op, args, value)
    # Initialize derivative (0 for scalars, zeros array for tensors)
    derivative = isa(value, Number) ? zero(value) : zeros(size(value))
    VectNode(op, args, value, derivative)
end

# Constructors for leaf nodes (inputs)
VectNode(value::Number)        = VectNode(nothing, Any[], value, zero(value))
VectNode(value::AbstractArray) = VectNode(nothing, Any[], value, zeros(size(value)))


# ==========================================================
# ===  OPERATOR OVERLOADING FOR GRAPH BUILDING            ===
# ==========================================================

# Overload broadcasted operators to work on VectNodes.
# This allows elementwise operations like tanh.(x), x .* y, etc.

function Base.broadcasted(op::Function, x::VectNode)
    result_val = broadcast(op, x.value)
    return VectNode(Symbol(op), Any[x], result_val)
end

function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    result_val = broadcast(op, x.value, y.value)
    tag = (op === (*)) ? :hadamard : Symbol(op)  # mark hadamard product separately
    return VectNode(tag, Any[x, y], result_val)
end

function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    result_val = broadcast(op, x.value, y)
    tag = (op === (*)) ? :hadamard : Symbol(op)
    return VectNode(tag, Any[x, y], result_val)
end

function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    result_val = broadcast(op, x, y.value)
    tag = (op === (*)) ? :hadamard : Symbol(op)
    return VectNode(tag, Any[x, y], result_val)
end

# Power operation for constant exponent
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{p}) where {p}
    result_val = x.value .^ p
    return VectNode(:^, Any[x, p], result_val)
end


# ==========================================================
# ===  BASIC ARITHMETIC OPERATORS (+, -, *, /, ^)         ===
# ==========================================================

# Addition and subtraction between nodes and numbers/arrays
Base.:+(x::VectNode, y::VectNode)      = VectNode(:+, Any[x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number)        = VectNode(:+, Any[x, y], x.value .+ y)
Base.:+(x::Number, y::VectNode)        = VectNode(:+, Any[x, y], x .+ y.value)
Base.:+(x::VectNode, y::AbstractArray) = VectNode(:+, Any[x, VectNode(y)], x.value .+ y)
Base.:+(x::AbstractArray, y::VectNode) = VectNode(:+, Any[VectNode(x), y], x .+ y.value)

Base.:-(x::VectNode, y::VectNode)      = VectNode(:-, Any[x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number)        = VectNode(:-, Any[x, y], x.value .- y)
Base.:-(x::Number, y::VectNode)        = VectNode(:-, Any[x, y], x .- y.value)
Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, Any[x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, Any[VectNode(x), y], x .- y.value)

# Multiplication — handles scalar, vector, and matrix cases
Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    result = if isa(xv, AbstractVector) && isa(yv, AbstractVector) && ndims(xv) == 1 && ndims(yv) == 1
        xv' * yv  # dot product
    else
        xv * yv   # standard matrix or scalar multiplication
    end
    VectNode(:*, Any[x, y], result)
end
Base.:*(x::VectNode, y::Number)        = VectNode(:*, Any[x, y], x.value * y)
Base.:*(x::Number, y::VectNode)        = VectNode(:*, Any[x, y], x * y.value)
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, Any[VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, Any[x, VectNode(y)], x.value * y)

# Division
Base.:/(x::VectNode, y::VectNode)      = VectNode(:/, Any[x, y], x.value / y.value)
Base.:/(x::VectNode, y::Number)        = VectNode(:/, Any[x, VectNode(y)], x.value / y)
Base.:/(x::Number, y::VectNode)        = VectNode(:/, Any[VectNode(x), y], x / y.value)

# Integer power
Base.:^(x::VectNode, n::Integer)       = Base.power_by_squaring(x, n)


# ==========================================================
# ===  NONLINEAR FUNCTIONS (ACTIVATIONS, LOG, EXP, SUM)   ===
# ==========================================================

# Elementwise ReLU and tanh
relu(x::VectNode) = VectNode(:relu, Any[x], max.(0, x.value))
tanh(x::VectNode) = VectNode(:tanh, Any[x], tanh.(x.value))

# Extend tanh for Dual (forward-mode differentiation)
function tanh(x::Dual)
    t = tanh(x.value)
    Dual(t, (1 - t) * (1 + t) * x.derivative)
end

# Other elementwise functions
Base.exp(x::VectNode) = VectNode(:exp, Any[x], exp.(x.value))
Base.log(x::VectNode) = VectNode(:log, Any[x], log.(x.value))
Base.sum(x::VectNode) = VectNode(:sum, Any[x], sum(x.value))


# ==========================================================
# ===  GRAPH UTILITIES                                    ===
# ==========================================================

# Topological sort (depth-first search)
# Ensures we process nodes from inputs to outputs in order.
function topo_sort(f::VectNode)
    visited = Set{VectNode}()
    order = VectNode[]
    function dfs(n::VectNode)
        if n ∉ visited
            push!(visited, n)
            for a in n.args
                isa(a, VectNode) && dfs(a)
            end
            push!(order, n)
        end
    end
    dfs(f)
    order
end

# Reset all derivatives to zero before backpropagation
function clear_derivatives!(nodes::Vector{VectNode})
    for n in nodes
        if isa(n.derivative, Number)
            n.derivative = zero(n.derivative)
        else
            n.derivative .= 0
        end
    end
end


# ==========================================================
# ===  REVERSE-MODE BACKPROPAGATION IMPLEMENTATION        ===
# ==========================================================

function backward!(f::VectNode)
    # 1️⃣ Sort nodes topologically
    nodes = topo_sort(f)

    # 2️⃣ Clear old derivatives
    clear_derivatives!(nodes)

    # 3️⃣ Initialize derivative of output to 1 (df/df = 1)
    f.derivative = isa(f.value, Number) ? one(f.value) : ones(size(f.value))

    # 4️⃣ Propagate gradients backward
    for node in Iterators.reverse(nodes)
        isnothing(node.op) && continue
        grad = node.derivative

        # --- Operation-specific gradient rules ---
        if node.op == :+
            # d(x + y) = dx + dy
            for a in node.args
                isa(a, VectNode) && (a.derivative .+= grad)
            end

        elseif node.op == :-
            # d(x - y) = dx - dy
            x, y = node.args
            isa(x, VectNode) && (x.derivative .+= grad)
            isa(y, VectNode) && (y.derivative .-= grad)

        elseif node.op == :hadamard
            # Elementwise multiplication
            x, y = node.args
            xv = isa(x, VectNode) ? x.value : x
            yv = isa(y, VectNode) ? y.value : y
            isa(x, VectNode) && (x.derivative .+= grad .* yv)
            isa(y, VectNode) && (y.derivative .+= grad .* xv)

        elseif node.op == :*
            # Matrix / vector / scalar multiplication
            x, y = node.args
            xv = isa(x, VectNode) ? x.value : x
            yv = isa(y, VectNode) ? y.value : y

            if isa(xv, Number) && isa(yv, Number)
                isa(x, VectNode) && (x.derivative += grad * yv)
                isa(y, VectNode) && (y.derivative += grad * xv)

            elseif isa(xv, AbstractVector) && isa(yv, AbstractVector)
                isa(x, VectNode) && (x.derivative .+= grad .* yv)
                isa(y, VectNode) && (y.derivative .+= grad .* xv)

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                isa(x, VectNode) && (x.derivative .+= grad * yv')
                isa(y, VectNode) && (y.derivative .+= xv' * grad)

            elseif isa(xv, AbstractArray) && isa(yv, Number)
                isa(x, VectNode) && (x.derivative .+= grad .* yv)
                isa(y, VectNode) && (y.derivative += sum(grad .* xv))

            elseif isa(xv, Number) && isa(yv, AbstractArray)
                isa(x, VectNode) && (x.derivative += sum(grad .* yv))
                isa(y, VectNode) && (y.derivative .+= grad .* xv)

            else
                error("Unhandled * case with types: $(typeof(xv)), $(typeof(yv))")
            end

        elseif node.op == :/
            # Quotient rule
            x, y = node.args
            xv = isa(x, VectNode) ? x.value : x
            yv = isa(y, VectNode) ? y.value : y
            isa(x, VectNode) && (x.derivative .+= grad ./ yv)
            isa(y, VectNode) && (y.derivative .-= grad .* xv ./ (yv .^ 2))

        elseif node.op == :^
            # Power rule
            arg, n = node.args
            upd = grad .* (n .* (arg.value .^ (n - 1)))
            arg.derivative .+= upd

        elseif node.op == :tanh
            # d(tanh(x)) = 1 - tanh^2(x)
            arg = node.args[1]
            upd = grad .* ((1 .- node.value) .* (1 .+ node.value))
            arg.derivative .+= upd

        elseif node.op == :relu
            # d(ReLU(x)) = 1 if x>=0 else 0
            arg = node.args[1]
            upd = grad .* float.(arg.value .>= 0)
            arg.derivative .+= upd

        elseif node.op == :exp
            arg = node.args[1]
            arg.derivative .+= grad .* node.value

        elseif node.op == :log
            arg = node.args[1]
            arg.derivative .+= grad ./ arg.value

        elseif node.op == :sum
            arg = node.args[1]
            arg.derivative .+= grad

        else
            error("Operation `$(node.op)` not supported in backward pass yet")
        end
    end

    return f
end


# ==========================================================
# ===  GRADIENT COMPUTATION API                          ===
# ==========================================================

# Computes gradient of f(x) with respect to x
function gradient!(f, g::Flatten, x::Flatten)
    # Wrap input components as differentiable VectNodes
    xnodes = Flatten(VectNode.(x.components))

    # Evaluate function with graph-building
    expr = f(xnodes)

    # Run reverse-mode backpropagation
    backward!(expr)

    # Extract derivatives into g
    @inbounds for i in eachindex(x.components)
        g.components[i] .= xnodes.components[i].derivative
    end
    g
end

# Convenience wrapper (allocates new gradient)
gradient(f, x) = gradient!(f, zero(x), x)


# ==========================================================
# ===  HESSIAN AND HESSIAN-VECTOR PRODUCT (SECOND ORDER) ===
# ==========================================================

# HVPExecutor structure wraps f and x for vectorized Hessian computation
struct HVPExecutor
    f::Function
    x::Flatten
end

# Compute flattened Hessian using forward-mode Hessian
function Base.vec(op::HVPExecutor)
    H = fwd_hessian(op.f, op.x)
    return vec(H)
end

# Interface functions
hvp(f, x::Flatten) = HVPExecutor(f, x)
const hessian = hvp  # Alias (same functionality)

end # module
