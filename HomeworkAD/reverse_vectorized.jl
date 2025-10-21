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
VectNode(x::Number) = VectNode(nothing, VectNode[], x, zero(x), zero(x))
VectNode(x::AbstractArray) = VectNode(nothing, VectNode[], x, zeros(size(x)), zeros(size(x)))

# Utility to create a VectNode with given op
function VectNode(op, args::Vector{VectNode}, value)
    derivative = isa(value, Number) ? zero(value) : zeros(size(value))
    tangent = isa(value, Number) ? zero(value) : zeros(size(value))
    return VectNode(op, args, value, derivative, tangent)
end

#-------------------------------
Base.length(x::VectNode) = length(x.value)
Base.size(x::VectNode) = size(x.value)  
Base.size(x::VectNode, dim) = size(x.value, dim)
Base.eltype(x::VectNode) = eltype(x.value)
Base.ndims(x::VectNode) = ndims(x.value)
Base.iterate(x::VectNode) = iterate(x.value)
Base.iterate(x::VectNode, state) = iterate(x.value, state)
Base.Broadcast.broadcastable(x::VectNode) = x

# -----------------------
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

function forward_tangent!(f::VectNode)
    sorted_nodes = topo_sort(f)
    
    for node in sorted_nodes
        if isnothing(node.op)
            continue

        elseif node.op == :+
            node.tangent = zero(node.value)
            for a in node.args
                # Handle both scalar and array addition
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
                # Handle unary minus if it occurs
                x = node.args[1]
                node.tangent = -x.tangent
            end

        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = x.tangent, y.tangent

            # Match ALL the cases from your reverse pass
            if isa(xv, Number) && isa(yv, Number)
                node.tangent = xt * yv + xv * yt
            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                # Handle both matrix multiplication and element-wise
                try
                    node.tangent = xt * yv + xv * yt
                catch
                    # Fallback to element-wise
                    node.tangent = xt .* yv .+ xv .* yt
                end
            elseif isa(xv, AbstractArray) && isa(yv, Number)
                node.tangent = xt .* yv .+ xv .* yt
            elseif isa(xv, Number) && isa(yv, AbstractArray)
                node.tangent = xt .* yv .+ xv .* yt
            else
                error("Unhandled * case: $(typeof(xv)), $(typeof(yv))")
            end

        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = x.tangent, y.tangent

            if isa(xv, Number) && isa(yv, Number)
                node.tangent = (xt * yv - xv * yt) / (yv^2)
            else
                node.tangent = (xt .* yv .- xv .* yt) ./ (yv .^ 2)
            end

        elseif node.op == :^
            # Handle both single arg (n=2) and two arg cases
            if length(node.args) == 1
                arg = node.args[1]
                n = 2
                xv = arg.value
                xt = arg.tangent
            else
                x, n_node = node.args
                n = n_node.value
                xv = x.value
                xt = x.tangent
            end
            
            if isa(xv, AbstractArray)
                node.tangent = n .* (xv .^(n-1)) .* xt
            else
                node.tangent = n * xv^(n-1) * xt
            end

        elseif node.op == :tanh
            arg = node.args[1]
            node.tangent = (1 .- node.value.^2) .* arg.tangent

        elseif node.op == :exp
            arg = node.args[1]
            node.tangent = node.value .* arg.tangent

        elseif node.op == :log
            arg = node.args[1]
            node.tangent = arg.tangent ./ arg.value

        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            node.tangent = relu_grad .* arg.tangent

        elseif node.op == :sum
            arg = node.args[1]
            node.tangent = sum(arg.tangent)

        else
            error("Op $(node.op) not implemented in forward_tangent!")
        end
    end
end
# -----------------------
# Compute gradient
# -----------------------
# Replace your gradient functions with this:
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


# -----------------------
# Compute full Hessian
# -----------------------
# Replace your entire Hessian computation with this:

function pushforward(f::Function, x::Flatten, tx::Vector)
    # Convert inputs to VectNodes with tangents
    x_nodes = [VectNode(xi) for xi in x.components]
    
    # Seed tangents with direction tx
    offset = 0
    for node in x_nodes
        n = length(node.value)
        node.tangent = reshape(tx[offset+1:offset+n], size(node.value))
        offset += n
    end
    
    x_flat = Flatten(x_nodes)
    
    # Compute function with forward pass
    result = f(x_flat)
    forward_tangent!(result)
    
    # Return tangents of output
    if isa(result, VectNode)
        return [result.tangent]
    else
        return [node.tangent for node in result.components]
    end
end

function jacobian(f::Function, x::Flatten)
    total_input_len = sum(length.(x.components))
    total_output_len = sum(length.(f(x).components))
    
    J = zeros(total_output_len, total_input_len)
    
    for j in 1:total_input_len
        # Create one-hot direction vector
        v = zeros(total_input_len)
        v[j] = 1.0
        
        # Compute jacobian column
        J_col = pushforward(f, x, v)
        J_col_vec = flatten_tangent(J_col)
        J[:, j] .= J_col_vec
    end
    
    return J
end

function hessian(f::Function, x::Flatten)
    # This matches the reference exactly: jacobian(z -> gradient(f, z), x)
    return jacobian(z -> gradient(f, z), x)
end

# Hessian-vector product
function hvp(f::Function, x::Flatten, v::Vector)
    return pushforward(z -> gradient(f, z), x, v)
end

end