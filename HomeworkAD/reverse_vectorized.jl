## Correct the implemention fo relu 
module VectReverse

mutable struct VectNode
    op::Union{Nothing, Symbol}
	args::Vector{VectNode} ## 1 D array of objects
	value::Any
	derivative::Any
end

#correctly broadcasting to minimise the errros 


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
#Base.broadcastable(x::VectNode) = Ref(x)

## intitalize the nodes derivatives to zero

#VectNode(op, args, value) = VectNode(op, args, value, zeros(size(value)))

## Define constructor not to worry about the values that are zero
VectNode(op, args, value) = VectNode(op, args, value, zeros(size(value)))

## Defining Constructors for the inputs
## For AbstractMatrix
VectNode(value::Number) = VectNode(nothing, VectNode[], value, zero(value))

VectNode(value::AbstractArray) = VectNode(nothing, VectNode[], value, zeros(size(value)))

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
## It does not 

#MethodError Julia does not know what this is 
#Base.:*(x::VectNode, y::VectNode) = VectNode(:* ,[x, y], x.value * y.value) 

# Overwriting global vector multiplication
# infinite recursion 
# because x * y inside calls the same method again

#Base.:*(x::AbstractVector, y::AbstractVector) = VectNode(:*, [x, y], x * y)  #No method exit for this combination or stackoverflow 

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

## ISLESS
#Base.isless(x::VectNode, y::Number) = isless(x.value, y)
#Base.isless(x::Number, y::VectNode) = isless(x, y.value)
#Base.isless(x::VectNode, y::VectNode) = isless(x.value, y.value)



## MAX - needed for relu operations
## Internally in relu we used max which uses is less so if x.val is a value then ok if not 
## We need to over the max operator accordingly as well.
## Not calls is less.
#Base.max(x::VectNode, y::Number) = VectNode(:relu, [x], max.(x.value, y))
#Base.max(x::Number, y::VectNode) = VectNode(:relu, [y], max.(x, y.value))
#Base.max(x::VectNode, y::VectNode) = VectNode(:relu, [x, y], max.(x.value, y.value))



#function relu(x::VectNode)

#	return VectNode(:relu,[x], max.(0.0, x.value))

#end
#--------------------------
#function relu(x::VectNode)
    # Elementwise max with 0
#    VectNode(:relu, [x], max.(0.0, x.value))
#end

#-------------------------

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
#--------------------------------------
#= function backward_relu!(node::VectNode)
    arg = node.args[1]
    
    # Debug prints
    println("=== RELU BACKWARD DEBUG ===")
    println("arg.value type: ", typeof(arg.value))
    println("arg.value size: ", size(arg.value))
    println("arg.value: ", arg.value)
    println("node.derivative type: ", typeof(node.derivative))
    println("node.derivative size: ", size(node.derivative))
    println("node.derivative: ", node.derivative)
    println("arg.derivative type: ", typeof(arg.derivative))
    println("arg.derivative size: ", size(arg.derivative))
    println("arg.derivative: ", arg.derivative)
    
    # Compute gradient elementwise (1 where x > 0, 0 elsewhere)
    relu_grad = float.(arg.value .> 0)
    println("relu_grad type: ", typeof(relu_grad))
    println("relu_grad size: ", size(relu_grad))
    println("relu_grad: ", relu_grad)
    
    # Extract scalar if 0-dimensional
    grad = ndims(node.derivative) == 0 ? node.derivative[] : node.derivative
    println("grad (after extraction) type: ", typeof(grad))
    println("grad: ", grad)
    
    # Accumulate gradient
    result = grad .* relu_grad
    println("result type: ", typeof(result))
    println("result: ", result)
    
    arg.derivative .+= result
    println("arg.derivative after update: ", arg.derivative)
    println("=========================\n")
end =#
function backward_relu!(node::VectNode)
    arg = node.args[1]
    
    # CRITICAL: Check where the OUTPUT (node.value) is > 0, not the input!
    relu_grad = float.(node.value .> 0)
    
    # Extract scalar if 0-dimensional
    grad = ndims(node.derivative) == 0 ? node.derivative[] : node.derivative
    
    # Accumulate gradient
    arg.derivative .+= grad .* relu_grad
end
#-------------------------------------

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
				args.derivative .+=  node.derivative
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
			
			x.derivative .+= node.derivative * 1
			y.derivative .+= -node.derivative * 1


		elseif node.op ==:*

			x, y = node.args
			if isa(x.value, Number) && isa(y.value, Number) ## scaler * scaler 
				x.derivative += node.derivative * y.value
				y.derivative += node.derivative * x.value
			
			elseif isa(x.value, AbstractArray) && isa(y.value, AbstractArray)
				## Some times node.derivative is a scaler which cannot be multipled directly 	
				
				## Here the derivative was represented not as scaler but zero dim (0 dim) array 
				## if 0d * vector throws error multiplied by scaler or vector not allowed in Julia

				grad = ndims(node.derivative) == 0 ? node.derivative[] : node.derivative

				if ndims(x.value) == 1 && ndims(y.value) == 1
					x.derivative .+= grad * y.value  ## vec * vector  alos scaler to vector 
					y.derivative .+= x.value * grad

				elseif ndims(x.value) == 2            ## matrix * matrix 
					 x.derivative .+= grad * y.value'	## this will work if grad is (m,n) (n,p) = m,p  bca multplication requires compatibles shapes
        			y.derivative .+= x.value' * grad
				else
					x.derivative .+= grad * y.value'  ##  for higher dims  > 2 v v
					y.derivative .+= x.value' * grad
				end
			else 
				if isa(x.value, AbstractArray)
					# x is arry, y is also scaler if both are vector or matrices then * 
					x.derivative .+= node.derivative .* y.value
            		y.derivative += sum(node.derivative .* x.value)
				
				end
				 # Mixed: x array, y scalar
				x.derivative .+= node.derivative .* y.value ## vector * scaler 
				y.derivative .+= node.derivative .* x.value
			end 
			
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
		
		elseif node.op ==:log
			arg = node.args[1]
			arg.derivative .+= node.derivative ./ arg.value
	
		elseif node.op ==:relu
			#arg = node.args[1]

			## Apply relu function
			#relu_grad = (arg.value .> 0) 
			#arg.derivative .+= node.derivative .* relu_grad
			backward_relu!(node)

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

