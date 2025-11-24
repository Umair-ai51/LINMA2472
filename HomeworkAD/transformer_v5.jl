module TinyTransformer1

using HTTP
using Random
using ..ForwardOverReverse: gradient, VectNode
import ..Flatten

const url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const text = String(HTTP.get(url).body)

const chars = sort(collect(Set(text)))
const vocab_size = length(chars)

const stoi = Dict(c => i for (i, c) in enumerate(chars))
const itos = Dict(i => c for (i, c) in enumerate(chars))

encode(s) = [stoi[c] for c in s]
const data_int = encode(text)

function get_batch(data::Vector{Int}, T::Int)
    i = rand(1:length(data) - T - 1)
    x = data[i:i+T-1]
    y = data[i+1:i+T]
    return x, y
end

struct TinyTransformer
    token   
    Wq     
    Wk      
    Wv      
    Wo      
    W1      
    b1      
    W2      
    b2      
    Wout    
end

function init_model(vocab::Int, d_model::Int)
    d_ff = 4 * d_model
    s() = 0.02 .* randn()

    M(m, n) = reshape([s() for _ in 1:m*n], m, n)
    V(n)    = reshape([s() for _ in 1:n], n)

    token = M(d_model, vocab)

    Wq = M(d_model, d_model)
    Wk = M(d_model, d_model)
    Wv = M(d_model, d_model)
    Wo = M(d_model, d_model)

    W1 = M(d_ff, d_model)
    b1 = V(d_ff)
    W2 = M(d_model, d_ff)
    b2 = V(d_model)

    Wout = M(vocab, d_model)

    return TinyTransformer(token, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Wout)
end

function flatten(m::TinyTransformer)
    components = Any[
        m.token,
        m.Wq, m.Wk, m.Wv, m.Wo,
        m.W1, m.b1, m.W2, m.b2,
        m.Wout,
    ]
    return Flatten(components)
end

function unflatten(θ::Flatten)
    c = θ.components
    @assert length(c) == 10 "Expected 10 components in Flatten for TinyTransformer"
    return TinyTransformer(
        c[1],
        c[2], c[3], c[4], c[5],
        c[6], c[7], c[8], c[9],
        c[10],
    )
end


function attention(x, m::TinyTransformer)

    Q = m.Wq * x                 
    K = m.Wk * x                 
    V = m.Wv * x                 

    d_model = size(Q.value, 1)
    T = size(Q.value, 2)

    scores = (transpose(Q) * K) / sqrt(d_model)

    ex = exp.(scores)
    A = ex ./ sum(ex)          

    out = V * transpose(A)
    return m.Wo * out           
end

import ..ForwardOverReverse: relu

function ffn(x, m::TinyTransformer)

    h = m.W1 * x .+ m.b1         
    h = relu(h)                  
    y = m.W2 * h .+ m.b2        
    return y
end


function forward(m::TinyTransformer, x_idx::Vector{Int})
    T = length(x_idx)
    d_model = size(m.token.value, 1)

    X = zeros(Float64, vocab_size, T)
    for t in 1:T
        X[x_idx[t], t] = 1.0
    end

    x = m.token * X

    a = attention(x, m)
    h = x + a

    h2 = h + ffn(h, m)

    logits = m.Wout * h2
    return logits
end

function cross_entropy(logits, y_idx::Vector{Int})
    V, T = size(logits.value)

    Y = zeros(Float64, V, T)
    for t in 1:T
        Y[y_idx[t], t] = 1.0
    end

    ex = exp.(logits)
    Z = sum(ex)
    P = ex ./ Z

    ce = sum((-Y) .* log.(P)) / T  
    return ce
end


const T_SEQ = 64

function loss_fn(θ::Flatten)
    m = unflatten(θ)   
    x, y = get_batch(data_int, T_SEQ)
    logits = forward(m, x)
    loss = cross_entropy(logits, y)
    return loss
end

function train(; d_model=32, num_steps=200, lr=1e-3)
    println("Initializing model...")
    model0 = init_model(vocab_size, d_model)
    θ = flatten(model0)

    for step in 1:num_steps
        g = gradient(loss_fn, θ)

        for i in eachindex(θ.components)
            θ.components[i] .-= lr .* g.components[i]
        end

        if step % 20 == 0
            θ_nodes = Flatten(VectNode.(θ.components))
            L = loss_fn(θ_nodes) 
            println("step $step  loss = ", L.value)
        end
    end

    return θ
end

end
