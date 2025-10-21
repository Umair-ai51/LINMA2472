module ForwardOverReverse

import ..Flatten

# -----------------------
# VectNode definition
# -----------------------
mutable struct VectNode
    op::Union{Nothing, Symbol}
    args::Vector{VectNode}
    value::Any
    derivative::Any
    tangent::Any
end

# Constructors
#VectNode(x::Number) = VectNode(nothing, VectNode[], x, zero(x), zero(x))
#VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x, zeros(size(x)), zeros(size(x)))


VectNode(x::Number) = VectNode(nothing, VectNode[], x, zero(x), zero(x))

#VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x, zeros(size(x)), zeros(size(x)))

# Utility to create a VectNode with given op
function VectNode(op, args::Vector{VectNode}, value)
    # ALWAYS Float64, never inherit type from value
    if isa(value, Number)
        derivative = 0.0
        tangent = 0.0
    else
        derivative = zeros(Float64, size(value))
        tangent = zeros(Float64, size(value))
    end
    return VectNode(op, args, value, derivative, tangent)
end


function VectNode(x::AbstractArray)
    # Derivatives are always floating point
    derivative = zeros(Float64, size(x))
    tangent = zeros(Float64, size(x))
    return VectNode(nothing, VectNode[], x, derivative, tangent)
end


#------------------------
# Broadcasting Support
# -----------------------

# For unary broadcasted operations like `tanh.(X)`
function Base.broadcasted(op::Function, x::VectNode)
    result_val = op.(x.value)
    return VectNode(Symbol(op), [x], result_val)
end

# For binary broadcasted operations `X .* Y` (both VectNode)
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    result = broadcast(op, x.value, y.value)
    return VectNode(Symbol(op), [x, y], result)
end

# For `X .* Y` where Y is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray, Number})
    result = broadcast(op, x.value, y)
    return VectNode(Symbol(op), [x, VectNode(y)], result)
end

# For `X .* Y` where X is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray, Number}, y::VectNode)
    result = broadcast(op, x, y.value)
    return VectNode(Symbol(op), [VectNode(x), y], result)
end

# For `x .^ 2` (literal power)
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
    result = x.value .^ y
    return VectNode(:^, [x], result)
end

# -----------------------
# Operator Overloads
# -----------------------

# Addition
Base.:+(x::VectNode, y::VectNode) = VectNode(:+, [x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number) = VectNode(:+, [x, VectNode(y)], x.value + y)
Base.:+(x::Number, y::VectNode) = VectNode(:+, [VectNode(x), y], x + y.value)
Base.:+(x::VectNode, y::AbstractArray) = VectNode(:+, [x, VectNode(y)], x.value .+ y)
Base.:+(x::AbstractArray, y::VectNode) = VectNode(:+, [VectNode(x), y], x .+ y.value)

# Subtraction
Base.:-(x::VectNode, y::VectNode) = VectNode(:-, [x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number) = VectNode(:-, [x, VectNode(y)], x.value - y)
Base.:-(x::Number, y::VectNode) = VectNode(:-, [VectNode(x), y], x - y.value)
Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, [x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, [VectNode(x), y], x .- y.value)

# Multiplication
Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    val = if isa(xv, AbstractVector) && isa(yv, AbstractVector)
        xv' * yv  # dot product
    else
        xv * yv
    end
    VectNode(:*, [x, y], val)
end
Base.:*(x::VectNode, y::Number) = VectNode(:*, [x, VectNode(y)], x.value * y)
Base.:*(x::Number, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, [x, VectNode(y)], x.value * y)
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)

# Division
Base.:/(x::VectNode, y::VectNode) = VectNode(:/, [x, y], x.value / y.value)
Base.:/(x::VectNode, y::Number) = VectNode(:/, [x, VectNode(y)], x.value / y)
Base.:/(x::Number, y::VectNode) = VectNode(:/, [VectNode(x), y], x / y.value)

# Power (only integer powers)
Base.:^(x::VectNode, n::Integer) = VectNode(:^, [x], x.value^n)

# Unary functions
Base.tanh(x::VectNode) = VectNode(:tanh, [x], tanh.(x.value))
Base.exp(x::VectNode) = VectNode(:exp, [x], exp.(x.value))
Base.log(x::VectNode) = VectNode(:log, [x], log.(x.value))
Base.sum(x::VectNode) = VectNode(:sum, [x], sum(x.value))

# ReLU activation
function relu(x::VectNode)
    return VectNode(:relu, [x], max.(0, x.value))
end

# -----------------------
# Topological Sort
# -----------------------
function topo_sort(f::VectNode)
    visited = Set{VectNode}()
    order = VectNode[]
    function dfs(n::VectNode)
        if n ∉ visited
            push!(visited, n)
            for a in n.args
                if isa(a, VectNode)
                    dfs(a)
                end
            end
            push!(order, n)
        end
    end
    dfs(f)
    return order
end

# -----------------------
# Reverse-mode backward
# -----------------------
function backward!(f::VectNode)
    sorted_nodes = topo_sort(f)
    f.derivative = 1.0
    for node in reverse(sorted_nodes)
        #if isa(node.derivative, BitArray)
        #    node.derivative = zeros(Float64, size(node.derivative))
        #end
        #if isa(node.tangent, BitArray)
        #    node.tangent = zeros(Float64, size(node.tangent))
        #end

        if isnothing(node.op)
            continue
        elseif node.op == :+
            for a in node.args
                a.derivative .+= node.derivative
            end
        elseif node.op == :-
            x, y = node.args
            x.derivative .+= node.derivative
            y.derivative .+= -node.derivative
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            
            if isa(xv, Number) && isa(yv, Number)
                x.derivative += grad * yv
                y.derivative += grad * xv

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                x.derivative .+= grad * yv'
                y.derivative .+= xv' * grad

            #= elseif isa(xv, AbstractArray) && isa(yv, Number)
                x.derivative .+= grad .* yv
                y.derivative += sum(grad .* xv)

            elseif isa(xv, Number) && isa(yv, AbstractArray)
                x.derivative += sum(grad .* yv)
                y.derivative .+= grad .* xv
             =#
            else
                error("Unhandled * case with types: $(typeof(xv)), $(typeof(yv))")
            end
        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            
            if isa(xv, Number) && isa(yv, Number)
                x.derivative += grad / yv
                y.derivative += grad * (-xv / yv^2)
            else
                x.derivative .+= grad ./ yv
                y.derivative .+= grad .* (-xv ./ yv.^2)
            end
        elseif node.op == :tanh
            arg = node.args[1]
            arg.derivative .+= node.derivative .* (1 .- node.value.^2)

        elseif node.op == :^
            arg = node.args[1]
            n = 2
            if isa(arg.value, AbstractArray)
                arg.derivative .+= node.derivative .* (n .* arg.value.^(n-1))
            else
                arg.derivative += node.derivative * (n * arg.value^(n-1))
            end

        elseif node.op == :exp
            arg = node.args[1]
            arg.derivative .+= node.derivative .* node.value

        elseif node.op == :log
            arg = node.args[1]
            arg.derivative .+= node.derivative ./ arg.value
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)

            arg.derivative .+= node.derivative .* relu_grad
        elseif node.op == :sum
            arg = node.args[1]
            arg.derivative .+= node.derivative
        else
            error("Op $(node.op) not implemented in backward!")
        end
    end
end


# -----------------------
# Forward pass for tangent (Hessian-vector product)
# -----------------------
# -----------------------
# Forward pass for tangent (Hessian-vector product) - FIXED
# -----------------------
# -----------------------
# Forward pass for tangent (Hessian-vector product) - FIXED
# -----------------------
# -----------------------
# Forward pass for tangent (Hessian-vector product)
# -----------------------
# Add these essential methods FIRST

function forward_on_backward!(f::VectNode)
    """
    This function differentiates the BACKWARD PASS.
    
    After backward! runs:
    - node.derivative contains ∂f/∂(node.value)
    
    This function computes:
    - node.tangent contains d/dv[∂f/∂(node.value)]
    
    where v is the perturbation direction seeded in the input nodes.
    
    For the INPUT nodes, node.tangent will contain the Hessian-vector product H*v.
    """
    
    sorted_nodes = topo_sort(f)
    
    # Process nodes in FORWARD order (same as backward pass, but differentiating each rule)
    for node in sorted_nodes
        if isnothing(node.op)
            continue

        elseif node.op == :+
            # Backward rule: for a in args: a.derivative += node.derivative
            # Differentiate: d/dv[a.derivative] = d/dv[node.derivative]
            # So: a.tangent (gradient tangent) gets contributions from node.tangent
            node.tangent = zero(node.value)
            for a in node.args
                if isa(node.tangent, Number) && isa(a.tangent, Number)
                    node.tangent += a.tangent
                else
                    node.tangent = node.tangent .+ a.tangent
                end
            end

        elseif node.op == :-
            if length(node.args) == 2
                x, y = node.args
                node.tangent = x.tangent - y.tangent
            else
                x = node.args[1]
                node.tangent = -x.tangent
            end

        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            
            # Here's what we need to understand:
            # We already ran backward!, so we know node.derivative (let's call it ḡ)
            # We also have x.tangent and y.tangent which tell us how x and y VALUES change with v
            # 
            # The backward rules were:
            #   x.derivative += ḡ * y
            #   y.derivative += x * ḡ  (or x' * ḡ for matrices)
            #
            # Now differentiate these rules w.r.t. the input perturbation v:
            #   d/dv[x.derivative] = d/dv[ḡ] * y + ḡ * d/dv[y]
            #                      = node.tangent * y + ḡ * y.tangent
            #
            # But wait - we're computing node.tangent HERE, and it should represent
            # how the PRIMAL output changes, which feeds into how downstream gradients change

            if isa(xv, Number) && isa(yv, Number)
                # Primal: z = x * y
                # Tangent of primal: dz/dv = dx/dv * y + x * dy/dv
                node.tangent = x.tangent * yv + xv * y.tangent

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                # Primal: Z = X * Y
                # Tangent: dZ/dv = dX/dv * Y + X * dY/dv
                node.tangent = x.tangent * yv + xv * y.tangent

            elseif isa(xv, AbstractArray) && isa(yv, Number)
                node.tangent = x.tangent .* yv .+ xv .* y.tangent

            elseif isa(xv, Number) && isa(yv, AbstractArray)
                node.tangent = x.tangent .* yv .+ xv .* y.tangent
            else
                error("Unhandled * case")
            end

        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = x.tangent, y.tangent

            if isa(xv, Number) && isa(yv, Number)
                # Primal: z = x / y
                # Tangent: dz/dv = (dx/dv * y - x * dy/dv) / y²
                node.tangent = (xt * yv - xv * yt) / (yv^2)
            else
                node.tangent = (xt .* yv .- xv .* yt) ./ (yv .^ 2)
            end

        elseif node.op == :^
            if length(node.args) == 1
                arg = node.args[1]
                n = 2  # hardcoded for x^2
                xv = arg.value
                xt = arg.tangent
            else
                x, n_node = node.args
                n = n_node.value
                xv = x.value
                xt = x.tangent
            end
            
            # Primal: y = x^n
            # Tangent: dy/dv = n * x^(n-1) * dx/dv
            if isa(xv, AbstractArray)
                node.tangent = n .* (xv .^(n-1)) .* xt
            else
                node.tangent = n * xv^(n-1) * xt
            end

        elseif node.op == :tanh
            # Primal: y = tanh(x)
            # Backward: x.derivative += node.derivative * (1 - y²)
            # 
            # For Hessian, we need to differentiate the backward rule:
            # The gradient ∂f/∂x = ḡ * (1 - tanh²(x))
            # Taking derivative w.r.t. input perturbation v:
            # d/dv[∂f/∂x] = d/dv[ḡ] * (1 - tanh²(x)) + ḡ * d/dv[1 - tanh²(x)]
            #              = node.tangent * (1 - y²) + ḡ * (-2*tanh(x)) * (∂tanh/∂x) * x.tangent
            #              = node.tangent * (1 - y²) + ḡ * (-2*y) * (1-y²) * x.tangent
            #
            # But for the PRIMAL tangent (forward mode on original function):
            arg = node.args[1]
            sech2 = 1 .- node.value.^2  # 1 - tanh²(x)
            
            # Primal tangent: d/dv[tanh(x)] = sech²(x) * dx/dv
            node.tangent = sech2 .* arg.tangent

        elseif node.op == :exp
            # Primal: y = exp(x)
            # Tangent: dy/dv = exp(x) * dx/dv = y * dx/dv
            arg = node.args[1]
            node.tangent = node.value .* arg.tangent

        elseif node.op == :log
            # Primal: y = log(x)
            # Tangent: dy/dv = (1/x) * dx/dv
            arg = node.args[1]
            node.tangent = arg.tangent ./ arg.value

        elseif node.op == :relu
            # Primal: y = max(0, x)
            # Tangent: dy/dv = (x >= 0) * dx/dv
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            node.tangent = relu_grad .* arg.tangent

        elseif node.op == :sum
            # Primal: y = sum(x)
            # Tangent: dy/dv = sum(dx/dv)
            arg = node.args[1]
            node.tangent = sum(arg.tangent)

        else
            error("Op $(node.op) not implemented!")
        end
    end
end


# -----------------------
# Compute gradient
# -----------------------
# Replace your gradient functions with this:

# ... all your VectNode definitions, operators, backward pass, etc. ...

# Gradient computation
# ======================================================================================
# CORRECT Forward-over-Reverse Implementation
# ======================================================================================
# The key: We must differentiate the BACKWARD PASS rules themselves!
function compute_value_tangents!(f::VectNode, value_tangents::Dict)
    """
    Standard forward-mode AD: compute how values change with input perturbation.
    Stores in value_tangents dict to keep separate from gradient tangents.
    """
    sorted_nodes = topo_sort(f)
    
    for node in sorted_nodes
        if isnothing(node.op)
            # Leaf - already seeded
            value_tangents[node] = node.tangent
            continue
        
        elseif node.op == :+
            vt = zero(node.value)
            for a in node.args
                vt = vt .+ value_tangents[a]
            end
            value_tangents[node] = vt
            
        elseif node.op == :-
            x, y = node.args
            value_tangents[node] = value_tangents[x] .- value_tangents[y]
            
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = value_tangents[x], value_tangents[y]
            
            if isa(xv, Number) && isa(yv, Number)
                value_tangents[node] = xt * yv + xv * yt
            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                value_tangents[node] = xt * yv + xv * yt
            else
                value_tangents[node] = xt .* yv .+ xv .* yt
            end
            
        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = value_tangents[x], value_tangents[y]
            
            if isa(xv, Number) && isa(yv, Number)
                value_tangents[node] = (xt * yv - xv * yt) / (yv^2)
            else
                value_tangents[node] = (xt .* yv .- xv .* yt) ./ (yv.^2)
            end
            
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            xv = arg.value
            xt = value_tangents[arg]
            
            if isa(xv, AbstractArray)
                value_tangents[node] = n .* (xv.^(n-1)) .* xt
            else
                value_tangents[node] = n * xv^(n-1) * xt
            end
            
        elseif node.op == :tanh
            arg = node.args[1]
            sech2 = 1 .- node.value.^2
            value_tangents[node] = sech2 .* value_tangents[arg]
            
        elseif node.op == :exp
            arg = node.args[1]
            value_tangents[node] = node.value .* value_tangents[arg]
            
        elseif node.op == :log
            arg = node.args[1]
            value_tangents[node] = value_tangents[arg] ./ arg.value
            
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            value_tangents[node] = relu_grad .* value_tangents[arg]
            
        elseif node.op == :sum
            arg = node.args[1]
            value_tangents[node] = sum(value_tangents[arg])
            
        else
            error("Op not implemented in value tangents!")
        end
    end
end


# Step 2: Differentiate the BACKWARD pass using value tangents

function gradient!(f, g::Flatten, x::Flatten)
    x_nodes = Flatten(VectNode.(x.components))
    expr = f(x_nodes)
    backward!(expr)
    for i in eachindex(x.components)
        g.components[i] .= x_nodes.components[i].derivative
    end
    return g
end

function gradient(f::Function, x::Flatten)
    g = zero(x)
    return gradient!(f, g, x)
end



function compute_gradient_tangents!(f::VectNode, value_tangents::Dict)
    """
    This differentiates the backward pass to compute how gradients change.
    Uses value_tangents computed in step 1.
    Stores result in node.tangent field (now representing gradient tangent).
    """
    sorted_nodes = topo_sort(f)
    
    # Initialize gradient tangents
    f.tangent = 0.0  # Output gradient is constant
    
    # Process in REVERSE order (differentiating backward pass)
    for node in reverse(sorted_nodes)
        if isnothing(node.op)
            continue
            
        elseif node.op == :+
            # Backward: a.derivative += node.derivative
            # Differentiate: d/dv[a.derivative] += d/dv[node.derivative]
            for a in node.args
                a.tangent .+= node.tangent
            end

        elseif node.op == :-
            x, y = node.args
            x.tangent .+= node.tangent
            y.tangent .-= node.tangent
            
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan_x = value_tangents[x]
            val_tan_y = value_tangents[y]
            
            # Backward was:
            #   x.derivative += grad * y_value  
            #   y.derivative += x_value * grad (or x' * grad for matrices)
            # Differentiate:
            #   d/dv[x.derivative] += d/dv[grad] * y + grad * d/dv[y]
            #   d/dv[y.derivative] += d/dv[x] * grad + x * d/dv[grad]
        
            if isa(xv, Number) && isa(yv, Number)
                x.tangent += grad_tan * yv + grad * val_tan_y
                y.tangent += val_tan_x * grad + xv * grad_tan
                
            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                # Backward was: x.derivative += grad * y', y.derivative += x' * grad
                x.tangent .+= grad_tan * yv' .+ grad * val_tan_y'
                y.tangent .+= val_tan_x' * grad .+ xv' * grad_tan
            else
                error("Unhandled case")
            end
            
        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan_x = value_tangents[x]
            val_tan_y = value_tangents[y]
            
            if isa(xv, Number) && isa(yv, Number)
                # Backward: x.derivative += grad/y, y.derivative += grad*(-x/y²)
                x.tangent += grad_tan/yv - grad*val_tan_y/(yv^2)
                y.tangent += grad_tan*(-xv/yv^2) + grad*(2*xv*val_tan_y/yv^3 - val_tan_x/yv^2)
            else
                x.tangent .+= grad_tan ./ yv .- grad .* val_tan_y ./ (yv.^2)
                y.tangent .+= grad_tan .* (-xv ./ yv.^2) .+ grad .* (2 .* xv .* val_tan_y ./ yv.^3 .- val_tan_x ./ yv.^2)
            end
            
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            xv = arg.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            # Backward: arg.derivative += grad * n * x^(n-1)
            if isa(xv, AbstractArray)
                arg.tangent .+= grad_tan .* (n .* xv.^(n-1)) .+ grad .* (n .* (n-1) .* xv.^(n-2)) .* val_tan
            else
                arg.tangent += grad_tan * (n * xv^(n-1)) + grad * (n * (n-1) * xv^(n-2)) * val_tan
            end
            
        elseif node.op == :tanh
            arg = node.args[1]
            y = node.value
            sech2 = 1 .- y.^2
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            # Backward: arg.derivative += grad * sech²(x)
            # d/dx[sech²(x)] = -2*tanh(x)*sech²(x)
            g_prime = -2 .* y .* sech2
            arg.tangent .+= grad_tan .* sech2 .+ grad .* g_prime .* val_tan
            
        elseif node.op == :exp
            arg = node.args[1]
            exp_val = node.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            # Backward: arg.derivative += grad * exp(x)
            arg.tangent .+= grad_tan .* exp_val .+ grad .* exp_val .* val_tan
            
        elseif node.op == :log
            arg = node.args[1]
            xv = arg.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            # Backward: arg.derivative += grad/x
            arg.tangent .+= grad_tan ./ xv .- grad .* val_tan ./ (xv.^2)
            
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            grad_tan = node.tangent
            
            # Backward: arg.derivative += grad * (x >= 0)
            arg.tangent .+= grad_tan .* relu_grad
            
        elseif node.op == :sum
            arg = node.args[1]
            arg.tangent .+= node.tangent
            
        else
            error("Op not implemented!")
    
        end
    end
end


# ======================================================================================
# Hessian-vector product
# ======================================================================================

function Hv(f::Function, x::Flatten, v::Vector)
    """
    Computes H(x) * v using forward-over-reverse.
    
    Algorithm:
    1. Build graph and evaluate f(x)
    2. Backward pass: compute gradients
    3. Forward pass on VALUES: compute how values change with perturbation v
    4. Backward pass on GRADIENTS: compute how gradients change using value tangents
    """
    
    # Step 1: Build graph
    x_nodes = [VectNode(xi) for xi in x.components]
    x_flat = Flatten(x_nodes)
    
    expr = f(x_flat)
    all_nodes = topo_sort(expr)
    
    # Step 2: Backward pass - compute gradients
    for node in all_nodes
        node.derivative = zero(node.value)
    end
    backward!(expr)
    
    # Step 3: Seed input tangents for value tangents and compute them
    for node in all_nodes
        node.tangent = zero(node.value)
    end
    
    offset = 0
    for node in x_nodes
        n = length(node.value)
        tangent_val = reshape(v[offset+1:offset+n], size(node.value))
        node.tangent = tangent_val
        offset += n
    end
    
    value_tangents = Dict{VectNode, Any}()
    compute_value_tangents!(expr, value_tangents)
    
    # Step 4: Reset tangents and compute gradient tangents
    for node in all_nodes
        node.tangent = zero(node.value)
    end
    
    compute_gradient_tangents!(expr, value_tangents)
    
    # Result is in x_nodes[i].tangent (now representing gradient tangent = H*v)
    return [node.tangent for node in x_nodes]
end


# ======================================================================================
# Helper and Hessian functions
# ======================================================================================

function flatten_tangent(tangents::Vector)
    v = Float64[]
    for t in tangents
        append!(v, vec(t))
    end
    return v
end

function hessian(f::Function, x::Flatten)
    total_len = sum(length.(x.components))
    H = zeros(total_len, total_len)

    for j in 1:total_len
        v = zeros(total_len)
        v[j] = 1.0
        
        Hv_col = Hv(f, x, v)
        Hv_vec = flatten_tangent(Hv_col)
        H[:, j] .= Hv_vec
    end

    return H
end

end
