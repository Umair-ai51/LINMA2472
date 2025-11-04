using HTTP
using Unicode
using Flux

# 1. Download dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
resp = HTTP.get(url)
text = String(resp.body)

# 2. Preprocess: lowercase, remove unwanted characters (optional)
text = lowercase(text)
# Keep letters, numbers, punctuation, and spaces
text = filter(c -> isletter(c) || isspace(c) || isdigit(c) || c in ['.',',','!','?',';',':','\'','"','-'], text)

# 3. Tokenize by whitespace
words = split(text)

# 4. Get unique words
unique_words = unique(words)

# 5. Display info
println("Total words: ", length(words))
println("Unique words: ", length(unique_words))
println("First 20 unique words: ", unique_words[1:20])

## Defining the structur fo reverse Ad

mutable struct trnsf_Node                  # unique node id
    op::Union{Nothing, Symbol}
    args::Vector{trnsf_Node}
    value::Any
    derivative::Any
    meta::Any

end

# -------------------------
# Graph printing helper
# -------------------------
function node_label(node::trnsf_Node)
    op = isnothing(node.op) ? "Leaf" : string(node.op)
    shape = try
        size(node.value)
    catch
        "(unknown size)"
    end
    meta_str = haskey(node.meta, :inds) ? " inds=$(node.meta[:inds])" : ""
    return "$op | shape=$shape$meta_str"
end

function print_graph(node::trnsf_Node; indent::Int=0, visited=Set{UInt}())
    nid = objectid(node)  # UInt
    indent_str = " " ^ indent

    if nid in visited
        println(indent_str, "↳ [shared node] ", node_label(node), " (id=$(nid))")
        return
    end
    push!(visited, nid)

    # Print this node
    println(indent_str, "Node id=$(nid): ", node_label(node))

    # Recursively print parents / args
    for arg in node.args
        print_graph(arg; indent=indent+4, visited=visited)
    end
end


# -------------------------
# Usage after forward pass:
# -------------------------
# pred = predict(first_seq, P)
# println("\n=== Expression graph (detailed) ===")
# print_graph(pred)
# println("\n=== Expression graph (tree) ===")
# print_graph_tree(pred)



## define constructors for operations
trnsf_Node(op,args, value) = trnsf_Node(op, args, value, zeros(size(value)),  Dict(:shape => size(value)))

## constructore for the leaf nodes 
trnsf_Node(x::AbstractArray) = trnsf_Node(nothing, trnsf_Node[], x, zeros(size(x)) ,Dict(:shape => size(x)))

## constructor for leaf node
trnsf_Node(x::Number) = trnsf_Node(nothing, trnsf_Node[], x, zeros(size(x)),  Dict(:shape => (1,)))


Base.:*(x::AbstractArray, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x * y)
Base.:*(x::Number, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x .* y )
Base.:*(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:*, [x, y], x.value * y.value)

Base.:*(x::Number, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x .* y )


Base.:+(x::AbstractArray, y::AbstractArray) = trnsf_Node(:+, [trnsf_Node(x), trnsf_Node(y)], x + y)
Base.:+(x::AbstractArray, y::trnsf_Node) = trnsf_Node(:+, [trnsf_Node(x), y], x + y.value)
Base.:+(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:+, [x, y], x.value + y.value)
Base.:+(x::trnsf_Node, y::AbstractVector) = trnsf_Node(:+, [x, trnsf_Node(y)], x.value .+ y)


Base.:-(x::trnsf_Node, y::AbstractArray) = trnsf_Node(:-, [x, trnsf_Node(y)], x.value .- y)

## Overloading the adjoint 
Base.:adjoint(x::trnsf_Node) = trnsf_Node(:adjoint, [x], x.value')
Base.sqrt(x::trnsf_Node)  = trnsf_Node(:sqrt, [x], sqrt.(x.value))
#Base.maximum(n::trnsf_Node, dims::Int) = maximum(n.value, dims= dims) 


Base.:/(x::trnsf_Node, y::Number) = trnsf_Node(:/ , [x, trnsf_Node(y)], x.value ./ y) 

relu(x) = max(0, x)

relu(x::trnsf_Node) = trnsf_Node(:relu, [x], relu.(x.value))
    
#Base.:transpose(x::trnsf_Node) = trnsf_Node(:transpose, [x], x.value')

## Overloading the base operator
Base.size(n::trnsf_Node) = size(n.value)
Base.size(n::trnsf_Node, dim::Int) = size(n.value, dim)
Base.length(n::trnsf_Node) = length(n.value)

#Base.getindex(n::trnsf_Node, inds...) = n.value[inds...]

##-------------
## The Problem We need to overload the index so we could get the index of 
## last token 
function Base.getindex(x::trnsf_Node, row_inds::Union{Colon, Int}, col_inds::Union{Colon, Int})
    # Create a new node representing the indexing
    new_node = trnsf_Node(:getindex, [x], x.value[row_inds, col_inds])
    # Store the indices in metadata for backward
    new_node.meta = Dict(:inds => (row_inds, col_inds))
    return new_node
end

##-------------
function softmax_node(scores::trnsf_Node)

    score_stable = scores.value .- maximum(scores.value, dims=2)
    exp_scores = exp.(score_stable)
    softmax_Scores = exp_scores ./ sum(score_stable, dims=2)

    return trnsf_Node(:softmax, [scores], softmax_Scores)

end

##--------------------------------
word2id = Dict{String, Int}()
id2word = Dict{Int, String}()

for (i, w) in enumerate(unique_words)
    word2id[w] = i
    id2word[i] = w
end

encoded_text = [word2id[w] for w in words]

block_size = 10
X = []
Y = []

# converting data into blocks

for i in 1:(length(encoded_text) - block_size)
    push!(X, encoded_text[i: i+block_size-1])
    push!(Y, encoded_text[i+1: i+block_size])

end

print(length(X))

print(size(X))
## define an embedding dims 

using LinearAlgebra
using Random

vocab_size = length(unique_words)

d_model = 64    

## randomly intialize weights and it uses column definition

W_embed = 0.01  * randn(d_model, vocab_size)

function embed(seq::Vector{Int})
    return hcat([W_embed.value[:, id] for id in seq]...)
end

function attention(Q, K, V)

    dk = size(K, 1)

    ## give similarity between keys and values
    ## where each row is one queyr 
    ## q1 the : k1 the, k2 cat 
    ## q2 cat : cat    
    ## how each token attend to another token
    print("\n The size of dk :", typeof(Q))
    print("\n The size of dfk:", size(K))
    
    #scores = (Q' * K) / sqrt(dk) ## Query and token 
    
    print("\n The size of K :", typeof(dk))
    print("\n The size of Q :", size(Q))
    print(dk)
    scores = (Q' * K) / sqrt(dk) ## Query and token 10x10

    ## convert the similarity to positive using exp
    ## subtract from max to avoid large value
    ## divide by sum
    ## for single word what is the prob of the next word 10,10
    print("\n Type of ", typeof(scores))
    
    scores = softmax_node(scores)

    #max_tok =  maximum(scores, dims=2)
    #scores = exp.(scores .- max_tok)
    #scores ./= sum(scores, dims=2)

    return V * scores 

end

function multihead_attention(X, num_heads=4)

    # no fo dim of embedding 
    #print("Type of X in MT",typeof(X))
    #print(size(X.value))
    ## extract the 64,10
    print("\n X in att_head", typeof(X))
    print("\n X in att_head", size(X))

    d_model = size(X, 1)

    print("\n d_model", typeof(d_model))
    print("\n d_model", size(d_model))

    d_head = div(d_model, num_heads)

    # intialize the 
    WQ = [0.01 * randn(d_head, d_model) for _ in 1:num_heads]
    WK = [0.01 * randn(d_head, d_model) for _ in 1:num_heads]
    WV = [0.01 * randn(d_head, d_model) for _ in 1:num_heads]
    print("Type of WQ ", typeof(WQ),"\n")
    print("Type of X ", typeof(X),"\n")

    print("size of WQ ", size(WQ[1].value),"\n")
    print("size of X ", size(X.value),"\n")

    heads = []
    
    for h in 1:num_heads
        Q = WQ[h] * X
        K = WK[h] * X
        V = WV[h] * X
        push!(heads, attention(Q,K,V))
    
    end

    print("Size of heads",size(heads))
    X_att = trnsf_Node(:concat, heads, vcat([h.value for h in heads]...))

    return X_att
end

function feed_forward(X)
    
    W1 = 0.01 * randn(128, size(X, 1))

    b1 = zeros(128)
    W2 = 0.01 * randn(size(X, 1), 128)
    b2 = zeros(size(X, 1))

    print("\n type  of W1 ",typeof(W1))
    print("\n type of X ",typeof(X))
    
    # Layer # 1
    l1 = W1 * X 
    l1 = l1 + b1 

    ## Activation
    l1 = relu(l1)
    
    ## Layer # 2
    l2 =  W2  * l1 

    y_pred = l2 + b2
    
    return y_pred

end

function Transformer_Block(X)

    X_att = multihead_attention(X)

    print("\n Size of X_att ",size(X_att))
    print("\n Size of X ",size(X))
    
    X = X + X_att

    print("Type of X is :", typeof(X))
    ## 
    X_ff = feed_forward(X)
    print("Size of X_ff", size(X_ff))
    X = X + X_ff
    print("Size of X after network : ", size(X))

    return X
end


W_out = 0.01 * randn(vocab_size, d_model)

function last_column(X::trnsf_Node)
    return X[:, size(X, 2)]
end

function predict(seq, P)

    X = embed(seq)
    print("Size of X",size(X))
    print("Size of vector ",typeof(X), "Size of PEM" ,typeof(P))
    print(size(X), size(P.value))
    X = X + P 

    X = Transformer_Block(X)
    print("\n Type of X", size(X),"\n")
    print("\n Type of W_out", size(W_out),"\n")

    ## (23000, 64) * (64, last token) taking the last token and multiplying it. 
    ## will give the probability distribution over 23000 words which then fed to softmax

    last_token = last_column(X)

    print("The shape of last token iis",size(last_token))
    logits = W_out * last_token

    return softmax_node(logits)  # predict next word based on last token

end


first_seq = X[1]

P = 0.01 * randn(d_model, block_size)

print("\n tye of P :" ,typeof(P))

pred = predict(first_seq, P)

println("size of the pred is : ",size(pred))

pred_id = argmax(pred.value)

println("Predicted next word ", id2word[pred_id])


