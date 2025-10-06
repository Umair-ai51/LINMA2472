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
## This will treat vectNode as scaler object and
## if the sum of that will give zero D vector which is an array stored a scaler (completely wrong)

#Base.broadcastable(x::VectNode) = Ref(x)

## intitalize the nodes derivatives to zero

function relu(x::VectNode)
    return VectNode(:relu, [x], max.(0, x.value))
end


function VectNode(op, args, value)
    if isa(value, Number)
        derivative = zero(value)     # scalar derivative
    else
        derivative = zeros(size(value))
    end
    return VectNode(op, args, value, derivative)
end
#-------------------------

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


		elseif node.op == :*
			x, y = node.args
			xv, yv = x.value, y.value
			grad = node.derivative
			
			println("shape of grad_ in before  extractions ", size(grad))
			println("shape of grad_ in after extractions ", size(grad))
			
			# scalar * scalar -> scalar
			if isa(xv, Number) && isa(yv, Number)
				x.derivative += grad * yv
				y.derivative += grad * xv

			# vector (n) dot vector (n) -> scalar: xv' * yv
			elseif isa(xv, AbstractVector) && isa(yv, AbstractVector) && ndims(xv) == 1 && ndims(yv) == 1
				x.derivative .+= grad * yv
				y.derivative .+= grad * xv

			# matrix * matrix (or matrix * vector) -> general matrix multiplication
			elseif isa(xv, AbstractArray) && isa(yv, AbstractArray)
				# Simply replace permutedims with transpose (')
				x.derivative .+= grad * yv'
				y.derivative .+= xv' * grad

			# mixed: x is array, y is scalar
			elseif isa(xv, AbstractArray) && isa(yv, Number)
				x.derivative .+= grad .* yv
				y.derivative += sum(grad .* xv)

			# mixed: x scalar, y array
			elseif isa(xv, Number) && isa(yv, AbstractArray)
				x.derivative += sum(grad .* yv)
				y.derivative .+= grad .* xv

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
		
		elseif node.op ==:log
			arg = node.args[1]
			arg.derivative .+= node.derivative ./ arg.value
	
		elseif node.op == :relu
			arg = node.args[1]

			## Apply relu function
			relu_grad = float.(arg.value .>= 0) 
			
			## -- Debug Prints ----
			println("node.value shape :" , size(node.value))
			println("node.value sample:", node.value[1:min(end, 5), 1:min(end, 5)])
			
			println("arg.derivate shape before update", size(arg.derivative))
			println("node.derivative shape: ", size(node.derivative))
			println("relu_grad shape", size(relu_grad))
			println("relu_grad sample", relu_grad[1:min(end, 5), 1:min(end, 5)])
			
			arg.derivative .+= node.derivative .* relu_grad
		
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

