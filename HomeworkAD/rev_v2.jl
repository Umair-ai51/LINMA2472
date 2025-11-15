## Correct the implemention fo relu 
module VectReverse

mutable struct VectNode
    op::Union{Nothing, Symbol}
	args::Vector{VectNode} ## 1 D array of objects
	value::Any
	derivative::Any
    second_derivative::Any
end

#correctly broadcasting to minimise the errros 

import Base: zero, one

# Define zero for VectNode
function Base.zero(x::VectNode)
    return VectNode(nothing, VectNode[], zero(x.value), zero(x.derivative), zero(x.second_derivative))
end

# Define one for VectNode (might be needed for some operations)
function Base.one(x::VectNode)
    return VectNode(nothing, VectNode[], one(x.value), zero(x.derivative), zero(x.second_derivative))
end

# Also define for type (not instance)
Base.zero(::Type{VectNode}) = VectNode(nothing, VectNode[], 0.0, 0.0, 0.0)
Base.one(::Type{VectNode}) = VectNode(nothing, VectNode[], 1.0, 0.0, 0.0)

##################----------------------------------------

# For `tanh.(X)`
function Base.broadcasted(op::Function, x::VectNode)
    ## apply the function elementwise
	result_val =  op.(x.value)
	
	#return the reuslt in the node.
	return VectNode(Symbol(op), [x], result_val)

end

# For `X .* Y`
function Base.broadcasted(op::Function, x::VectNode, y::VectNode)
	result = broadcast(op, x.value, y.value)
	return VectNode(Symbol(op), [x, y], result)
end


# For `X .* Y` where `Y` is a constant
function Base.broadcasted(op::Function, x::VectNode, y::Union{AbstractArray,Number})
		result = x.value .* y
		return VectNode(Symbol(op), [x, y], result)
end

# For `X .* Y` where `X` is a constant
function Base.broadcasted(op::Function, x::Union{AbstractArray,Number}, y::VectNode)
		result = x .* y.value
		return VectNode(Symbol(op), [x, y], result)
end

# For `x .^ 2`
function Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::VectNode, ::Val{y}) where {y}
	# Base.broadcasted(^, x, y)
	result = x.value .^ y
	return VectNode(:^, [x], result)  # or whatever structure you use
end


## by default julisa does not know how broadcast the vect node if we get a scaler node
## Throw the arrow of length 
## This will treat vectNode as scaler object and
## if the sum of that will give zero D vector which is an array stored a scaler (completely wrong)

#Base.broadcastable(x::VectNode) = Ref(x)

## intitalize the nodes derivatives to zero

#function relu(x::VectNode)
#    return VectNode(:relu, [x], max.(0, x.value))
#end

## Node that are created as sum of some other nodes

## Error : MethodError: no method matching Main.VectReverse.VectNode(::Symbol, ::Vector{Main.VectReverse.VectNode}, ::Matrix{Float64})
## Meaning no method exist for handling middles or operation of nodes.
function VectNode(op, args, value)
    if isa(value, Number)
        derivative = zero(value)     # scalar derivative
        second_derivative = zero(value)
    else
        derivative = zeros(size(value))
        second_derivative = zeros(size(value))
    end
    return VectNode(op, args, value, derivative, second_derivative)

end

#-------------------------

## Defining Constructors for the inputs Nodes with operation nothing 

## For Scalers
VectNode(value::Number) = VectNode(nothing, VectNode[], value, zero(value), zero(value),)


## For matrices or vectors  
VectNode(value::AbstractArray) = VectNode(nothing, VectNode[], value,zeros(size(value)) ,zeros(size(value)))


## Defining the operations for the forward pass

Base.:+(x::VectNode, y::VectNode) = VectNode(:+, [x, y], x.value + y.value)
Base.:+(x::VectNode, y::Number) = VectNode(:+, [x, y], x.value + y)
Base.:+(x::Number, y::VectNode) = VectNode(:+, [x, y], x + y.value)


Base.:-(x::VectNode, y::VectNode) = VectNode(:- ,[x, y], x.value - y.value)
Base.:-(x::VectNode, y::Number) = VectNode(:-, [x, y], x.value - y)
Base.:-(x::Number, y::VectNode) = VectNode(:-, [x, y], x - y.value)

## if we don't store abstract array in the same format
##  as node Julia will not be able to store it.


Base.:-(x::VectNode, y::AbstractArray) = VectNode(:-, [x, VectNode(y)], x.value .- y)
Base.:-(x::AbstractArray, y::VectNode) = VectNode(:-, [VectNode(x), y], x .- y.value)

# Vector *  Vector Take transpose Other wise dimensionality mismathc 

## Error of conversion 
## Error of dimensionality

#MethodError Julia does not know what this is 
#Base.:*(x::VectNode, y::VectNode) = VectNode(:* ,[x, y], x.value * y.value) 

#= Overwriting global vector multiplication
	infinite recursion 
	because x * y inside calls the same method again
	Base.:*(x::AbstractVector, y::AbstractVector) = VectNode(:*, [x, y], x * y)  #No method exit for this combination or stackoverflow 
 =#

#A::Matrix * B::Vector   # matrix–vector multiplication ✅
#A::Vector' * B::Vector  # dot product (row vector × column vector) ✅
#A::Vector * B'          # outer product (column × row) ✅

## Here the * AbstractVector are called internally with values from VectorNode So  

Base.:*(x::VectNode, y::VectNode) = begin
    xv, yv = x.value, y.value
    result = if isa(xv, AbstractVector) && isa(yv, AbstractVector)

        xv' * yv             # dot product
    else
        xv * yv              # normal matrix/number multiplication
    end
    VectNode(:*, [x, y], result)
end  
##############----------------------------
import Base: convert

# Allow conversion from any type to VectNode
function Base.convert(::Type{VectNode}, x::Number)
    return VectNode(x)
end

function Base.convert(::Type{VectNode}, x::AbstractArray)
    return VectNode(x)
end

function Base.convert(::Type{VectNode}, x::VectNode)
    return x
end


##############----------------------------

# Vector * Number 
Base.:*(x::VectNode, y::Number) = VectNode(:*, [x, y], x.value * y)
Base.:*(x::Number, y::VectNode) = VectNode(:*, [x, y], x * y.value)

# Vector * AbstractArray "Transpose" other wise dimensionlity mismatch (10,17, 10,21)
## Getting the the error of conversion from array to vector node to store it as a vector 
Base.:*(x::AbstractArray, y::VectNode) = VectNode(:*, [VectNode(x), y], x * y.value)
Base.:*(x::VectNode, y::AbstractArray) = VectNode(:*, [x, VectNode(y)], x.value * y)


Base.:/(x::VectNode, y::VectNode) = VectNode(:/ ,[x, y], x.value / y.value)

Base.:/(x::VectNode, y::Number) = VectNode(:/, [x, VectNode(y)], x.value / y)
Base.:/(x::Number, y::VectNode) = VectNode(:/, [VectNode(x), y], x / y.value)


Base.:^(x::VectNode, n::Integer) = Base.power_by_squaring(x, n)
#Base.:^(x::VectNode, n::Integer) = VectNode(:^, [x], x .^ n)

Base.tanh(x::VectNode) = VectNode(:tanh, [x], tanh.(x.value))

Base.exp(x::VectNode) = VectNode(:exp, [x], exp(x.value))
Base.log(x::VectNode) = VectNode(:log, [x], log(x.value))

## Defining sum
Base.sum(x::VectNode) = VectNode(:sum, [x], sum(x.value))


function topo_sort(f::VectNode)
	
	visited = Set{VectNode}()
	topological_order = VectNode[]

	## for each element in the node iterative recusively to 
	## store elements in acending order
	function _topo_sort__(node::VectNode)
		
			if node ∉ visited
				push!(visited, node)
				
				## checking for the further arguements if they exist
				for args in node.args
					## call the function recusively to find the elemnt that is null
					_topo_sort__(args)
				end
				## find the last node push it
				## Juli has the ability to mutate external varaibles
		
				push!(topological_order, node)
			end
	end

	_topo_sort__(f)
	return topological_order
	
end

# We assume `Flatten` has been defined in the parent module.
# If this fails, run `include("/path/to/Flatten.jl")` before
# including this file.
import ..Flatten

function backward!(f::VectNode)
	

	## call topological sort on the function
	sorted_nodes = topo_sort(f)

	f.derivative = 1.0

	## Iterate in reverse mmode
	## The value from the priovus node is also taken into account

	for node in reverse(sorted_nodes)

		if isnothing(node.op)
			continue 

		elseif node.op ==:+
			for args in node.args
				if isa(args.derivative, Number)
					args.derivative +=  node.derivative
				else 
					args.derivative .+= node.derivative
				end
			end
		
		## A scaler output will give a scaler gradient
		## The aim is to broad cast the single scaler to inputs
		## This will get constributions for all the elements
		## When the previous gradient is vector but the current is scaler
		
		#elseif node.op ==:+
		#	for args in node.args
		#		args.deriavtive .+= sum(node.deriavtive) .* ones(size(args.value)) 
		#	end
		

		elseif node.op ==:-
			x, y = node.args
		
			if isa(x.derivative, Number)
				x.derivative += node.derivative * 1
			else
				x.derivative .+= node.derivative * 1
			end

			if isa(y.derivative, Number)
				y.derivative += -node.derivative * 1
			else
				y.derivative .+= -node.derivative * 1
			end

		elseif node.op == :*
			x, y = node.args
			xv, yv = x.value, y.value
			grad = node.derivative
			
			#println("shape of grad_ in before  extractions ", size(grad))
			#println("shape of grad_ in after extractions ", size(grad))
			
			# scalar * scalar -> scalar
			if isa(xv, Number) && isa(yv, Number)
				x.derivative += grad * yv
				y.derivative += grad * xv

			## Wrong -------XXXXXXXXXXX------
			# vector (n) dot vector (n) -> scalar: xv' * yv  
			## if grad is scaler 
			#elseif isa(xv, AbstractVector) && isa(yv, AbstractVector) && ndims(xv) == 1 && ndims(yv) == 1
			#	x.derivative .+= grad * yv
			#	y.derivative .+= grad * xv

			# matrix * matrix (or matrix * vector) -> general matrix multiplication
			elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
				# Simply replace permutedims with transpose (')
				x.derivative .+= grad * yv'
				y.derivative .+= xv' * grad
				
			# mixed: x is array, y is scalar 
			## X = [1,2,3,4] And W = 2
			#elseif isa(xv, AbstractArray) && isa(yv, Number)
			#	x.derivative .+= grad .* yv
			#	y.derivative += sum(grad .* xv)

			# mixed: x scalar, y array
			#elseif isa(xv, Number) && isa(yv, AbstractArray)
			#	x.derivative += sum(grad .* yv)
			#	y.derivative .+= grad .* xv

			else
				error("Unhandled * case with types: $(typeof(xv)), $(typeof(yv))")
			end
		 #--------------------------
		elseif node.op ==:/
			## two case L = x/y
			## if L is differentieted wrt x = 1/Y
			x, y = node.args
			
			if isa(x.derivative, Number)
				x.derivative += node.derivative .* 1 ./ y.value
			else
				x.derivative .+= node.derivative .* 1 ./ y.value
			end
			## if L is differentieted wrt x = dl/dy = -x/Y^-2
			## Since computing mse and I have a scaler value cannot use broadcasting 
			if isa(y.derivative, Number)

				y.derivative += node.derivative .* (-x.value ./ y.value^2)
			else
				y.derivative .+= node.derivative .* (-x.value ./ y.value^2)
			end


		elseif node.op ==:tanh
			arg = node.args[1]
			if isa(arg.derivative, Number)
				arg.derivative += node.derivative .* (1  .- arg.value.^2)
			else
				arg.derivative .+= node.derivative .* (1 .- node.value.^2)	
			end

		elseif node.op ==:^
			arg = node.args[1]
			n = 2
			arg.derivative .+= node.derivative .* (n * arg.value.^ (n-1))
		
		elseif node.op ==:log ## dlog(x)  = 1/x
			arg = node.args[1]
			arg.derivative .+= node.derivative ./ arg.value 
	
		elseif node.op == :relu
			arg = node.args[1]

			## Apply relu function
			#relu_grad = Float64.(arg.value .>= 0) 
			
			## -- Debug Prints ----
			#println("node.value shape :" , size(node.value))
			#println("node.value sample:", node.value[1:min(end, 5), 1:min(end, 5)])
			
			#println("arg.derivate shape before update", size(arg.derivative))
			#println("node.derivative shape: ", size(node.derivative))
			#println("relu_grad shape", size(relu_grad))
			#println("relu_grad sample", relu_grad[1:min(end, 5), 1:min(end, 5)])
			
			#arg.derivative .+= node.derivative .* relu_grad

			if isa(arg.value, Number)
				#relu_grad = arg.value >= 0 ? 1.0 : 0.0

				arg.derivative += node.derivative * relu_grad
			else
				relu_grad = Float64.(arg.value .>= 0)
				arg.derivative .+= node.derivative .* relu_grad
			end
		
		elseif node.op ==:exp
			arg = node.args[1]
			arg.derivative .+=  node.derivative * node.value
		
		elseif node.op ==:sum
			arg = node.args[1]
			arg.derivative .+= node.derivative
		else
			error("Operation `$(f.op)` not supported yet")
		end
	end
end

function backward_second_order!(f::VectNode)
    sorted_nodes = topo_sort(f)
    f.second_derivative = 0.0

    for node in reverse(sorted_nodes)
        if isnothing(node.op)
            continue
        elseif node.op == :+
            for args in node.args
				if isa(args.second_derivative, Number)
                	args.second_derivative += node.second_derivative
				else
					args.second_derivative .+= node.second_derivative
			
				end
			end
        elseif node.op == :-
            x, y = node.args
			if isa(x.second_derivative,Number)

				x.second_derivative += node.second_derivative
				y.second_derivative += node.second_derivative
			else
				x.second_derivative .+= node.second_derivative
				y.second_derivative .+= node.second_derivative
			
			end
        elseif node.op == :*
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            grad2 = node.second_derivative

            if isa(xv, Number) && isa(yv, Number)
                x.second_derivative += grad2 * yv + grad * y.derivative
                y.second_derivative += grad2 * xv + grad * x.derivative
        
            elseif isa(xv, AbstractMatrix) && isa(yv, AbstractMatrix)
                x.second_derivative .+= grad2 * yv' .+ grad .* y.derivative'
                y.second_derivative .+= xv' * grad2 .+ x.derivative .* grad' 
            end
            
        elseif node.op == :/
            x, y = node.args
            xv, yv = x.value, y.value
            grad = node.derivative
            grad2 = node.second_derivative
            
            if isa(xv, Number) && isa(yv, Number)
                x.second_derivative += grad2 / yv + grad * (-y.derivative / yv^2)
                y.second_derivative += grad2 * (-xv / yv^2) + grad * (-x.derivative / yv^2 - 2 * xv * y.derivative / yv^3)
            else
                x.second_derivative .+= grad2 ./ yv .+ grad .* (-y.derivative ./ yv.^2)
                y.second_derivative .+= grad2 .* (-xv ./ yv.^2) .+ grad .* (-x.derivative ./ yv.^2 .- 2 .* xv .* y.derivative ./ yv.^3)
            end
        
        elseif node.op == :tanh
            arg = node.args[1]
            val = arg.value
            d1_tanh = 1 .- val.^2
            d2_tanh = -2 .* val .* d1_tanh
            arg.second_derivative .+= node.second_derivative .* d1_tanh .+ node.derivative .* d2_tanh .* arg.derivative
        
        elseif node.op == :exp
            arg = node.args[1]
            exp_val = node.value
            arg.second_derivative .+= node.second_derivative .* exp_val .+ node.derivative .* exp_val .* arg.derivative
        
        elseif node.op == :log
            arg = node.args[1]
            val = arg.value
            d1_log = 1 ./ val
            d2_log = -1 ./ val.^2
            arg.second_derivative .+= node.second_derivative .* d1_log .+ node.derivative .* d2_log .* arg.derivative
        
        elseif node.op == :^
            arg = node.args[1]
            n = 2
            val = arg.value
            d1_pow = n .* val.^(n-1)
            d2_pow = n .* (n-1) .* val.^(n-2)
            arg.second_derivative .+= node.second_derivative .* d1_pow .+ node.derivative .* d2_pow .* arg.derivative
        
        elseif node.op == :sum
            arg = node.args[1]
            arg.second_derivative .+= node.second_derivative
        
        elseif node.op == :relu
            arg = node.args[1]
            # ReLU second derivative is 0 (piecewise linear)
            # Do nothing
        
        else 
            error("Operation `$(node.op)` not supported in backward_second_order!")
        end
    end
end

function hessian(f, x::AbstractVector)
    n = length(x)
    h = zeros(n, n)

    for i in 1:n
        x_nodes = [VectNode(x[j]) for j in 1:n]
        
        expr = f(x_nodes)

        expr.derivative = 1.0
        backward!(expr)

        saved_first_derivs = [copy(x_nodes[j].derivative) for j in 1:n]
        
		expr.second_derivative = 0.0

		for node in x_nodes
            node.second_derivative = 0.0
        end

        x_nodes[i].derivative = 1.0

        backward_second_order!(expr)

        for j in 1:n
            h[i, j] = x_nodes[j].second_derivative
        end
    end
    return h
end
function hessian(f, x::Flatten)
    # Flatten all components into a vector
    x_vec = reduce(vcat, vec.(x.components))
    n = length(x_vec)
    h = zeros(n, n)

    # Create nodes
    x_nodes_components = []
    idx = 1
    for comp in x.components
        len = length(comp)
        vals = [VectNode(x_vec[idx + j - 1]) for j in 1:len]
        push!(x_nodes_components, reshape(vals, size(comp)))
        idx += len
    end
    x_nodes_flat = Flatten(x_nodes_components)

    # Compute Hessian
    expr = f(x_nodes_flat)
    backward!(expr)

    for i in 1:n
        for k in eachindex(x_nodes_components)
            for node in x_nodes_components[k]
				if isa(node.second_derivative, Number)
				    node.second_derivative = 0.0
				else
    				node.second_derivative .= 0.0
				end
			end
        end

        # Set derivative 1 for the i-th node
        idx = 1
        for k in eachindex(x_nodes_components)
            len = length(x_nodes_components[k])
            if i >= idx && i < idx + len
					x_nodes_components[k][i - idx + 1].derivative = 1.0
                break
            end
            idx += len
        end

        backward_second_order!(expr)

        idx = 1
        for k in eachindex(x_nodes_components)
            for node in x_nodes_components[k]
                h[i, idx] = node.second_derivative
                idx += 1
            end
        end

        # Reset derivative
        for k in eachindex(x_nodes_components)
            for node in x_nodes_components[k]
				#println("node.value type: ", typeof(node.value))
        		#println("node.value size: ", size(node.value))  # For arrays
        		#println("node.derivative type: ", typeof(node.derivative))
        	
			if isa(node.derivative, AbstractArray)	
            	#println("node.derivative size: ", size(node.derivative))
        	end
               
			end
        end
    end
    return h
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

gradient(f, x) = gradient!(f, zero(x), x)

end

