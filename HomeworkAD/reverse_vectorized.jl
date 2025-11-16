module ForwardOverReverse

import ..Flatten

# -----------------------
# VectNode definition
# -----------------------
mutable struct VectNode
    op::Union{Nothing, Symbol}
    args::Vector{VectNode} # vector of nodes [x, y] but their value could be matrices inputs are defined as vectors 
    value::Any
    derivative::Any
    tangent::Any
end

VectNode(x::Number) = VectNode(nothing, VectNode[], x, zero(x), zero(x))

## we want the forward pass to smoothly other wise in comparision we get bool values
function VectNode(op, args::Vector{VectNode}, value)
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
    # Always use Float64 for derivatives and tangents, if input is Bool/BitArray
    derivative = zeros(Float64, size(x))
    tangent = zeros(Float64, size(x))
    return VectNode(nothing, VectNode[], float.(x), derivative, tangent)
end

#------------------------
# Broadcasting Support
# -----------------------

function Base.broadcasted(op::Function, x::VectNode)
    result_val = op.(x.value)
    return VectNode(Symbol(op), [x], result_val)
end

function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
    result = broadcast(op, x.value, y.value)
    return VectNode(Symbol(op), [x, y], result)
end

function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray, Number})
    result = broadcast(op, x.value, y)
    return VectNode(Symbol(op), [x, VectNode(y)], result)
end

function Base.broadcasted(op::Function, x::Union{AbstractArray, Number}, y::VectNode)
    result = broadcast(op, x, y.value)
    return VectNode(Symbol(op), [VectNode(x), y], result)
end

function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
    result = broadcast(^, x.value, y)
    return VectNode(:^, [x], result)
end

# -----------------------
# Operator Overloads
# -----------------------

###
# Transformer
###


Base.transpose(x::VectNode) = VectNode(:transpose, [x], x.value')
Base.:+(x::VectNode, y::VectNode) = VectNode(:+, [x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number) = VectNode(:+, [x, VectNode(y)], x.value + y)
Base.:+(x::Number, y::VectNode) = VectNode(:+, [VectNode(x), y], x + y.value)
Base.:+(x::VectNode, y::AbstractArray) = VectNode(:+, [x, VectNode(y)], x.value .+ y)
Base.:+(x::AbstractArray, y::VectNode) = VectNode(:+, [VectNode(x), y], x .+ y.value)

Base.:-(x::VectNode, y::VectNode) = VectNode(:-, [x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number) = VectNode(:-, [x, VectNode(y)], x.value - y)
Base.:-(x::Number, y::VectNode) = VectNode(:-, [VectNode(x), y], x - y.value)
Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, [x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, [VectNode(x), y], x .- y.value)

Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    val =  xv * yv
    VectNode(:*, [x, y], val)
end

Base.:*(x::VectNode, y::Number) = VectNode(:*, [x, VectNode(y)], x.value * y)
Base.:*(x::Number, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, [x, VectNode(y)], x.value * y)
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)

Base.:/(x::VectNode, y::VectNode) = VectNode(:/, [x, y], x.value / y.value)
Base.:/(x::VectNode, y::Number) = VectNode(:/, [x, VectNode(y)], x.value / y)
Base.:/(x::Number, y::VectNode) = VectNode(:/, [VectNode(x), y], x / y.value)

Base.:^(x::VectNode, n::Integer) = VectNode(:^, [x], x.value^n)

Base.tanh(x::VectNode) = VectNode(:tanh, [x], tanh.(x.value))
Base.exp(x::VectNode) = VectNode(:exp, [x], exp.(x.value))
Base.log(x::VectNode) = VectNode(:log, [x], log.(x.value))
Base.sum(x::VectNode) = VectNode(:sum, [x], sum(x.value))

function relu(x::VectNode)
    return VectNode(:relu, [x], max.(0, x.value))
end

# Create a VectNode representing transpose(A) (i.e. A')
function Base.transpose(x::VectNode)
    return VectNode(:transpose, [x], x.value')
end

# Create a VectNode representing adjoint(A) (alias of transpose here for real arrays)
function Base.adjoint(x::VectNode)
    return VectNode(:adjoint, [x], x.value')
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
# Reverse-mode backward (Standard gradient computation)
# -----------------------
function backward!(f::VectNode)
    sorted_nodes = topo_sort(f)
    f.derivative = 1.0
    
    for node in reverse(sorted_nodes)
        if isnothing(node.op)
            continue

        # ----------------- +
        elseif node.op == :+
            for a in node.args
                # Cas scalaire
                if isa(a.derivative, Number) && isa(node.derivative, Number)
                    a.derivative += node.derivative

                # Même shape => propag direct
                elseif size(a.derivative) == size(node.derivative)
                    a.derivative .+= node.derivative

                # Cas biais : vecteur + matrice (broadcast sur la 2e dim)
                elseif ndims(a.derivative) == 1 && ndims(node.derivative) == 2 &&
                       length(a.derivative) == size(node.derivative, 1)
                    # On somme sur la dimension "temps" / colonnes
                    a.derivative .+= sum(node.derivative, dims = 2)[:]

                else
                    error("Broadcast gradient for + not implemented for shapes " *
                          "$(size(a.derivative)) and $(size(node.derivative))")
                end
            end

        # ----------------- -
        elseif node.op == :-
            x, y = node.args
            x.derivative .+= node.derivative
            y.derivative .+= -node.derivative

        # ----------------- *
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative

            if isa(xv, Number) && isa(yv, Number)
                # scalaire * scalaire
                x.derivative += grad * yv
                y.derivative += grad * xv

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
                # Cas 1 : produit élément par élément (broadcast .*)
                if size(node.value) == size(xv) == size(yv)
                    # z = x .* y  => dL/dx += grad .* y ; dL/dy += grad .* x
                    x.derivative .+= grad .* yv
                    y.derivative .+= grad .* xv
                else
                    # Cas 2 : vrai produit matriciel z = x * y
                    # x : (m×k), y : (k×n), grad : (m×n)
                    x.derivative .+= grad * yv'
                    y.derivative .+= xv' * grad
                end

            else
                error("Unhandled * case with types: $(typeof(xv)), $(typeof(yv))")
            end

        # ----------------- /
        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            
            if isa(xv, Number) && isa(yv, Number)
                x.derivative += grad / yv
                y.derivative += grad * (-xv / yv^2)
            end

        # ----------------- tanh
        elseif node.op == :tanh
            arg = node.args[1]
            arg.derivative .+= node.derivative .* (1 .- node.value.^2)

        # ----------------- ^
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            if isa(arg.value, AbstractArray)
                arg.derivative .+= node.derivative .* (n .* arg.value.^(n-1))
            else
                arg.derivative += node.derivative * (n * arg.value^(n-1))
            end

        # ----------------- exp
        elseif node.op == :exp
            arg = node.args[1]
            arg.derivative .+= node.derivative .* node.value

        # ----------------- log
        elseif node.op == :log
            arg = node.args[1]
            arg.derivative .+= node.derivative ./ arg.value

        # ----------------- relu
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            arg.derivative .+= node.derivative .* relu_grad

        # ----------------- sum
        elseif node.op == :sum
            arg = node.args[1]
            arg.derivative .+= node.derivative
        else
            error("Op $(node.op) not implemented in backward!")
        end
    end
end




# ======================================================================================
# OPTIMIZED: Single forward pass computes BOTH value tangents AND gradient tangents!
# ======================================================================================

function forward_hessian_vector!(f::VectNode)
    """
    Single forward pass that computes both:
    1. Value tangents: d/dv[node.value] (stored temporarily in value_tangents dict)
    2. Gradient tangents: d/dv[node.derivative] (stored in node.tangent)
    
    This is the OPTIMIZED version - only ONE forward pass!
    
    Prerequisites: 
    - backward!(f) must have been called (node.derivative filled)
    - Input nodes must have node.tangent seeded with direction v
    """
    sorted_nodes = topo_sort(f)
    ## storing value tangents in the dictionary why??

    value_tangents = Dict{VectNode, Any}()
    
    # Initialize value tangents for leaf nodes from their tangent field
    for node in sorted_nodes
        
        if isnothing(node.op)
            value_tangents[node] = node.tangent
        end
    end
    
    # Single forward pass computes BOTH value and gradient tangents
    for node in sorted_nodes
        if isnothing(node.op)
            # Leaf node - value tangent already set above
            continue
        end
        
        # ============================================================
        # For each operation, compute the value tangent
        # ============================================================

        if node.op == :+
            # Value tangent: sum of input value tangents
            vt = zero(node.value)
            for a in node.args
                vt = vt .+ value_tangents[a]
            end
            value_tangents[node] = vt
            
        elseif node.op == :-
            x, y = node.args
            ## value_tangents[x] gives the tangent of the node 
            value_tangents[node] = value_tangents[x] .- value_tangents[y]
            
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            xt, yt = value_tangents[x], value_tangents[y]
            
            # Value tangent: product rule
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
            
            # Value tangent: quotient rule
            if isa(xv, Number) && isa(yv, Number)
                value_tangents[node] = (xt * yv - xv * yt) / (yv^2)
            end
            
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            xv = arg.value
            xt = value_tangents[arg]
            
            # Value tangent: power rule
            if isa(xv, AbstractArray)
                value_tangents[node] = n .* (xv.^(n-1)) .* xt
            else
                value_tangents[node] = n * xv^(n-1) * xt
            end
            
        elseif node.op == :tanh
            arg = node.args[1]
            sech2 = 1 .- node.value.^2
            ## dtanh = 1 - node.value^2
            ## sech^2 .* value_tangents (1-tanh(x) * d(x))
            value_tangents[node] = sech2 .* value_tangents[arg]
            
        elseif node.op == :exp
            arg = node.args[1]
            value_tangents[node] = node.value .* value_tangents[arg]
            
        elseif node.op == :log
            arg = node.args[1]
            ## dlog(x) =   1/x * d(x)
            value_tangents[node] = value_tangents[arg] ./ arg.value
            
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            value_tangents[node] = relu_grad .* value_tangents[arg]
            
        elseif node.op == :sum
            arg = node.args[1]
            value_tangents[node] = sum(value_tangents[arg])
        
        elseif node.op == :transpose
            arg = node.args[1]
            # value_tangent of x' is (value_tangent of x)'
            value_tangents[node] = value_tangents[arg]'

            
        else
            error("Op not implemented in forward pass!")
        end
    end
    
    # ============================================================
    # Now compute gradient tangents in REVERSE order
    # This is the "differentiate the backward pass" step
    # ============================================================
    
    # Initialize: output gradient is constant
    for node in sorted_nodes
        node.tangent = zero(node.derivative)
    end
    
    sorted_nodes[end].tangent = 0.0
    
    for node in reverse(sorted_nodes)
        if isnothing(node.op)
            continue
            
        elseif node.op == :+
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
            
            # Differentiate backward rule: x.derivative += grad * y
            ## Differentiate 2nd rule: d(grad * y) ==> 2nd order * y2 +  
            if isa(xv, Number) && isa(yv, Number)
                x.tangent += grad_tan * yv + grad * val_tan_y
                y.tangent += val_tan_x * grad + xv * grad_tan

            elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
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
                ## the first term differentiates gradient and second term differentiates constant because d(grad/y) => 
                ## 2nd order grad/constant(y) - grad * d(y)/y^2 according to the or what
                x.tangent += grad_tan/yv - grad*val_tan_y/(yv^2)
                y.tangent += grad_tan*(-xv/yv^2) + grad*(2*xv*val_tan_y/yv^3 - val_tan_x/yv^2)
            else
                x.tangent .+= grad_tan ./ yv .- grad .* val_tan_y ./ (yv.^2)
                ## product rule with quotient rule 
                y.tangent .+= grad_tan .* (-xv ./ yv.^2) .+ grad .* (2 .* xv .* val_tan_y ./ yv.^3 .- val_tan_x ./ yv.^2)
            end
            
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            xv = arg.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            if isa(xv, AbstractArray)
                arg.tangent .+= grad_tan .* (n .* xv.^(n-1)) .+ grad .* (n .* (n-1) .* xv.^(n-2)) .* val_tan
            else
                ## grad * n* x^n-1 = > grad * conn + grad * n(n-1) (power rule again with respect ot the input )
                arg.tangent += grad_tan * (n * xv^(n-1)) + grad * (n * (n-1) * xv^(n-2)) * val_tan
            end
            
        elseif node.op == :tanh
            arg = node.args[1]
            y = node.value
            sech2 = 1 .- y.^2
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            g_prime = -2 .* y .* sech2
            ## grad_tan .* sech2 + grad d(sech2x) . val_tan(x)  
            arg.tangent .+= grad_tan .* sech2 .+ grad .* g_prime .* val_tan
            
        elseif node.op == :exp
            arg = node.args[1]
            exp_val = node.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            arg.tangent .+= grad_tan .* exp_val .+ grad .* exp_val .* val_tan
            
        elseif node.op == :log
            arg = node.args[1]
            xv = arg.value
            grad = node.derivative
            grad_tan = node.tangent
            val_tan = value_tangents[arg]
            
            arg.tangent .+= grad_tan ./ xv .- grad .* val_tan ./ (xv.^2)
            
        elseif node.op == :relu
            arg = node.args[1]
            relu_grad = float.(arg.value .>= 0)
            grad_tan = node.tangent
            
            arg.tangent .+= grad_tan .* relu_grad
            
        elseif node.op == :sum
            arg = node.args[1]
            arg.tangent .+= node.tangent

        elseif node.op == :transpose
            arg = node.args[1]
            # gradient tangent propagates through transpose the same way
            arg.tangent .+= node.tangent'

            
        else
            error("Op not implemented in gradient tangent!")
        end
    end
end

# ======================================================================================
# Hessian-vector product (OPTIMIZED VERSION)
# ======================================================================================

function Hv(f::Function, x::Flatten, v::Vector)
    """
    Computes H(x) * v using forward-over-reverse.
    
    OPTIMIZED Algorithm:
    1. Build graph and evaluate f(x)
    2. Backward pass: compute gradients
    3. Single forward pass: compute value tangents AND gradient tangents together
    """
    
    # Step 1: Build graph
    ## just using x we make the whole expression graph 
    x_nodes = [VectNode(xi) for xi in x.components]
    x_flat = Flatten(x_nodes)
    expr = f(x_flat)
    
    # Step 2: Backward pass - compute gradients
    all_nodes = topo_sort(expr)

    for node in all_nodes
        node.derivative = zero(node.value)
    end
    backward!(expr)
    
    # Step 3: Seed input tangents with v which will set [1,0,0,0,0,0,0,0] => tangent_value x = [1,0; 0,0], y = [0,0;0,0]
    offset = 0
    for node in x_nodes
        n = length(node.value)
        tangent_val = reshape(v[offset+1:offset+n], size(node.value))
        print(tangent_val)
        node.tangent = tangent_val
        offset += n
    end
    
    print("\n DONE \n")
    
    # Single optimized forward pass!
    forward_hessian_vector!(expr)
    
    # Result is in x_nodes[i].tangent (gradient tangent = H*v)
    for node in x_nodes
        print(node.tangent) 
    end
    return [node.tangent for node in x_nodes]
end

# ======================================================================================
# Gradient computation
# ======================================================================================

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
    
    ## each component has two inputs x and y
    # take the total sum to get the lengt
    total_len = sum(length.(x.components))
    ## intialize the hessian for these inputs 
    H = zeros(total_len, total_len)
 

    for j in 1:total_len
        v = zeros(total_len)
        v[j] = 1.0
        #println("Seed vector v = ", v)
        
        Hv_col = Hv(f, x, v)
        ## for first pass if X =[x11 x12 x13 x13], W = [W11, W12, W13, W14] => 
        ## v = [1,0,0,0.....] X[11] W[11] then X[12] W[12] X[13] W[13] for X11 we get W11 
        Hv_vec = flatten_tangent(Hv_col)
        H[:, j] .= Hv_vec
    end

    return H
end

end