LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))

## First order

#include(joinpath(@__DIR__, "fvf.jl"))


#run_gradient_tests(Forward.gradient, VectReverse.gradient)

include(joinpath(@__DIR__, "reverse_vectorized.jl"))

run_gradient_tests(Forward.gradient, ForwardOverReverse.gradient)


## Second order
# We only test `hessian` and not `hvp` but if `hessian` is implemented
# by reusing `hvp`, this is testing both at the same time.

include(joinpath(@__DIR__, "reverse_vectorized.jl"))

run_gradient_tests(Forward.hessian, ForwardOverReverse.hessian, hessian = true)


include(joinpath(@__DIR__, "G.jl"))
using .TinyTransformer1

θ = TinyTransformer1.train(num_steps = 200)   
m = TinyTransformer1.unflatten(θ)   