module VectReverse

mutable struct VectNode
    op::Union{Nothing, Symbol}
    args::Vector{VectNode}
    value::Any
    derivative::Any
end

# ReLU forward pass
function relu(x::VectNode)
    result = max.(0, x.value)
    return VectNode(:relu, [x], result)
end

# Broadcasted operations
function Base.broadcasted(op::Function, x::VectNode)
    result_val = op.(x.value)
    return VectNode(Symbol(op), [x], result_val)
end

function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    result = broadcast(op, x.value, y.value)
    return VectNode(Symbol(op), [x, y], result)
end

function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
    result = x.value .* y
    return VectNode(Symbol(op), [x, y], result)
end

function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
    result = x .* y.value
    return VectNode(Symbol(op), [x, y], result)
end

function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
    result = x.value .^ y
    return VectNode(:^, [x], result)
end

# Initialize derivatives
function VectNode(op, args, value)
    if isa(value, Number)
        derivative = zero(value)
    else
        derivative = zeros(size(value))
    end
    return VectNode(op, args, value, derivative)
end

# Constructors
VectNode(value::Number) = VectNode(nothing, VectNode[], value, zero(value))
VectNode(value::AbstractArray) = VectNode(nothing, VectNode[], value, zeros(size(value)))

# Forward pass operations
Base.:+(x::VectNode, y::VectNode) = VectNode(:+, [x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number) = VectNode(:+, [x, y], x.value + y)
Base.:+(x::Number, y::VectNode) = VectNode(:+, [x, y], x + y.value)

Base.:-(x::VectNode, y::VectNode) = VectNode(:-, [x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number) = VectNode(:-, [x, y], x.value - y)
Base.:-(x::Number, y::VectNode) = VectNode(:-, [x, y], x - y.value)
Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, [x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, [VectNode(x), y], x .- y.value)

Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    result = if isa(xv, AbstractVector) && isa(yv, AbstractVector)
        xv' * yv
    else
        xv * yv
    end
    VectNode(:*, [x, y], result)
end

Base.:*(x::VectNode, y::Number) = VectNode(:*, [x, y], x.value * y)
Base.:*(x::Number, y::VectNode) = VectNode(:*, [x, y], x * y.value)
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, [x, VectNode(y)], x.value * y)

Base.:/(x::VectNode, y::VectNode) = VectNode(:/, [x, y], x.value / y.value)
Base.:/(x::VectNode, y::Number) = VectNode(:/, [x, VectNode(y)], x.value / y)
Base.:/(x::Number, y::VectNode) = VectNode(:/, [VectNode(x), y], x / y.value)

Base.:^(x::VectNode, n::Integer) = Base.power_by_squaring(x, n)

Base.tanh(x::VectNode) = VectNode(:tanh, [x], tanh.(x.value))
Base.exp(x::VectNode) = VectNode(:exp, [x], exp(x.value))
Base.log(x::VectNode) = VectNode(:log, [x], log(x.value))
Base.sum(x::VectNode) = VectNode(:sum, [x], sum(x.value))

# Topological sort
function topo_sort(f::VectNode)
    visited = Set{VectNode}()
    topological_order = VectNode[]

    function _topo_sort__(node::VectNode)
        if node ∉ visited
            push!(visited, node)
            for args in node.args
                _topo_sort__(args)
            end
            push!(topological_order, node)
        end
    end

    _topo_sort__(f)
    return topological_order
end

import ..Flatten

"""Helper to flatten gradient results (from test file)"""
function flatten_gradient(g)
    if isa(g, Float64)
        return [g]
    elseif hasproperty(g, :components)
        return reduce(vcat, vec.(g.components))
    elseif isa(g, AbstractArray) && !isa(g, Vector)
        return reduce(vcat, flatten_gradient.(g))
    else
        return vec(g)
    end
end

function backward!(f::VectNode)
    sorted_nodes = topo_sort(f)
    f.derivative = 1.0

    for node in reverse(sorted_nodes)
        if isnothing(node.op)
            continue

        elseif node.op == :+
            for args in node.args
                args.derivative .+= node.derivative
            end

        elseif node.op == :-
            x, y = node.args
            x.derivative .+= node.derivative * 1
            y.derivative .+= -node.derivative * 1

        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative

            if isa(xv, Number) && isa(yv, Number)
                x.derivative += grad * yv
                y.derivative += grad * xv

            elseif isa(xv, AbstractVector) && isa(yv, AbstractVector) && ndims(xv) == 1 && ndims(yv) == 1
                x.derivative .+= grad * yv
                y.derivative .+= grad * xv

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                x.derivative .+= grad * yv'
                y.derivative .+= xv' * grad

            elseif isa(xv, AbstractArray) && isa(yv, Number)
                x.derivative .+= grad .* yv
                y.derivative += sum(grad .* xv)

            elseif isa(xv, Number) && isa(yv, AbstractArray)
                x.derivative += sum(grad .* yv)
                y.derivative .+= grad .* xv
            end

        elseif node.op == :/
            x, y = node.args
            if isa(x.derivative, Number)
                x.derivative += node.derivative .* 1 ./ y.value
            else
                x.derivative .+= node.derivative .* 1 ./ y.value
            end

            if isa(y.derivative, Number)
                y.derivative += node.derivative .* (-x.value ./ y.value^2)
            else
                y.derivative .+= node.derivative .* (-x.value ./ y.value^2)
            end

        elseif node.op == :tanh
            arg = node.args[1]
            if isa(arg.derivative, Number)
                arg.derivative += node.derivative .* (1 .- arg.value.^2)
            else
                arg.derivative .+= node.derivative .* (1 .- node.value.^2)
            end

        elseif node.op == :^
            arg = node.args[1]
            n = 2
            arg.derivative .+= node.derivative .* (n * arg.value.^ (n-1))

        elseif node.op == :log
            arg = node.args[1]
            arg.derivative .+= node.derivative ./ arg.value

        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            arg.derivative .+= node.derivative .* relu_grad

        elseif node.op == :exp
            arg = node.args[1]
            arg.derivative .+= node.derivative * node.value

        elseif node.op == :sum
            arg = node.args[1]
            arg.derivative .+= node.derivative

        else
            error("Operation `$(node.op)` not supported yet")
        end
    end
end

function gradient!(f, g::Flatten, x::Flatten)
    x_nodes = Flatten(VectNode.(x.components))
    expr = f(x_nodes)
    backward!(expr)
    for i in eachindex(x.components)
        g.components[i] .= x_nodes.components[i].derivative
    end
    return g
end

"""Gradient for vector inputs"""
function gradient!(f, g::Vector, x::Vector)
    # Convert vector to Flatten-like structure
    x_node = VectNode(x)
    expr = f(x_node)
    backward!(expr)
    g .= x_node.derivative
    return g
end

function gradient(f, x::Flatten)
    return gradient!(f, zero(x), x)
end

function gradient(f, x::Vector)
    g = zeros(length(x))
    return gradient!(f, g, x)
end

# ============================================
# HESSIAN IMPLEMENTATION: H(f) = J(∇f)
# ============================================

"""
    hessian(f, x::AbstractArray)

Compute the Hessian matrix H(f) = J(∇f) at point x.
This computes the Jacobian of the gradient:
- Takes the gradient of f as a vector-valued function
- Computes the Jacobian of that gradient (which is the Hessian)

For a scalar function f: ℝⁿ → ℝ, returns an n×n matrix where:
H[i,j] = ∂²f/∂x[i]∂x[j]

Uses numerical differentiation (finite differences) to compute the Jacobian.
"""
function hessian(f, x::AbstractArray)
    n = length(x)
    H = zeros(Float64, n, n)
    
    # Step size for finite differences
    ε = 1e-8
    
    # Compute each row of the Hessian
    # H[i,j] = ∂²f/∂x[i]∂x[j] = ∂/∂x[j](∂f/∂x[i])
    for j in 1:n
        # Perturb x in the j-th direction
        x_plus = copy(x)
        x_plus[j] += ε
        
        x_minus = copy(x)
        x_minus[j] -= ε
        
        # Compute gradients at perturbed points
        grad_plus = gradient(f, x_plus)
        grad_minus = gradient(f, x_minus)
        
        # Flatten gradients to vectors
        grad_plus_vec = flatten_gradient(grad_plus)
        grad_minus_vec = flatten_gradient(grad_minus)
        
        # Central difference approximation for each row
        for i in 1:n
            H[i, j] = (grad_plus_vec[i] - grad_minus_vec[i]) / (2ε)
        end
    end
    
    return H
end

"""
    hessian(f, x::Flatten)

Compute the Hessian matrix H(f) = J(∇f) for Flatten objects.
"""
function hessian(f, x::Flatten)
    n_total = sum(length.(x.components))
    H = zeros(Float64, n_total, n_total)
    
    ε = 1e-7
    
    for j in 1:n_total
        # Perturb in j-th direction
        x_plus = deepcopy(x)
        x_minus = deepcopy(x)
        
        # Find which component and index
        idx = 1
        for comp_idx in eachindex(x.components)
            comp_len = length(x.components[comp_idx])
            if j <= idx + comp_len - 1
                local_idx = j - idx + 1
                x_plus.components[comp_idx][local_idx] += ε
                x_minus.components[comp_idx][local_idx] -= ε
                break
            end
            idx += comp_len
        end
        
        # Compute gradients
        grad_plus = gradient(f, x_plus)
        grad_minus = gradient(f, x_minus)
        
        # Flatten and compute differences
        grad_plus_vec = flatten_gradient(grad_plus)
        grad_minus_vec = flatten_gradient(grad_minus)
        
        for i in 1:n_total
            H[i, j] = (grad_plus_vec[i] - grad_minus_vec[i]) / (2ε)
        end
    end
    
    return H
end

"""Helper function to convert Flatten object to vector"""
function Flatten_to_vec(x::Flatten)
    result = Float64[]
    for component in x.components
        append!(result, vec(component))
    end
    return result
end

"""Helper function to convert vector to Flatten object"""
function vec_to_Flatten(v::Vector, template::Flatten)
    new_flatten = deepcopy(template)
    idx = 1
    for i in eachindex(new_flatten.components)
        comp_len = length(new_flatten.components[i])
        new_flatten.components[i] .= v[idx:idx+comp_len-1]
        idx += comp_len
    end
    return new_flatten
end

"""Handle VectNode input by extracting value"""
function vec_to_Flatten(v::VectNode, template::Flatten)
    return vec_to_Flatten(v.value, template)
end

"""
    hessian!(result::AbstractArray, f, x::AbstractArray)

In-place version of hessian().
Computes H(f) = J(∇f) and stores result in the provided matrix.
"""
function hessian!(result::AbstractArray, f, x::AbstractArray)
    n = length(x)
    ε = 1e-8
    
    for j in 1:n
        x_plus = copy(x)
        x_plus[j] += ε
        
        x_minus = copy(x)
        x_minus[j] -= ε
        
        grad_plus = gradient(f, x_plus)
        grad_minus = gradient(f, x_minus)
        
        for i in 1:n
            result[i, j] = (grad_plus[i] - grad_minus[i]) / (2ε)
        end
    end
    
    return result
end

"""
    hessian!(result::AbstractArray, f, x::Flatten)

In-place Hessian computation for Flatten objects.
"""
function hessian!(result::AbstractArray, f, x::Flatten)
    x_vec = Flatten_to_vec(x)
    return hessian!(result, f, x_vec)
end

end