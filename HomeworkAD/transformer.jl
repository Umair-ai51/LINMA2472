using HTTP
using Unicode
using Flux
using Statistics

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

Base.:+(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:+, [x, y], x.value .+ y.value)

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


###########

function analyze_graph(node::trnsf_Node; indent::Int=0, visited=Set{UInt}())
    nid = objectid(node)
    indent_str = " " ^ indent

    # If already visited, print reference and stop recursion
    if nid in visited
        println(indent_str, "↳ [shared node] ", node_label(node), " (id=$(nid))")
        return 0
    end
    push!(visited, nid)

    # Print this node
    println(indent_str, "Node id=$(nid): ", node_label(node))

    # Recursively analyze arguments (children)
    total = 1  # count this node
    for arg in node.args
        total += analyze_graph(arg; indent=indent + 4, visited=visited)
    end

    # Only the top-level call should report the final count
    if indent == 0
        println("\n=== Summary ===")
        println("Total unique nodes in graph: ", length(visited))
    end

    return total
end




###########
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
    softmax_Scores = exp_scores ./ sum(exp_scores, dims=2)

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

W_embed = trnsf_Node(randn(d_model, vocab_size))


function embed(seq::Vector{Int})
    X_emb = hcat([W_embed.value[:, id] for id in seq]...)
    return trnsf_Node(:embed, [W_embed], X_emb)
end

function attention(Q::trnsf_Node, K::trnsf_Node, V::trnsf_Node; mask=false)

    dk = size(K, 1)

    ## give similarity between keys and values
    ## where each row is one queyr 
    ## q1 the : k1 the, k2 cat 
    ## q2 cat : cat    
    ## how each token attend to another token
    print("\n The size of Q :", typeof(Q))
    print("\n The size of Q:", size(Q))


    print("\n The size of K :", typeof(K))
    print("\n The size of K:", size(K))

    print("\n The size of V :", typeof(V))
    print("\n The size of V:", size(V))

  
    print(dk)
    scores = (Q' * K) / sqrt(dk) ## Query and token 10x10

    ## convert the similarity to positive using exp
    ## subtract from max to avoid large value
    ## divide by sum
    ## for single word what is the prob of the next word 10,10
    print("\n The size of scores:", size(scores))
    print("\n The type of scores:", typeof(scores))
    
    if mask 
        seq_len = size(scores, 2)
        mask_matrix = triu(ones(seq_len, seq_len), 1) .* -1e9
        scores = trnsf_Node(:masked_scores, [trnsf_Node(scores.value)], scores.value .+ mask_matrix)
    end

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
    WQ = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
    WK = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
    WV = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
    
    print("Type of WQ ", typeof(WQ),"\n")
    print("Type of X ", typeof(X),"\n")

    print("size of WQ ", size(WQ[1].value),"\n")
    print("size of X ", size(X.value),"\n")

    heads = []
    
    for h in 1:num_heads
        Q = WQ[h] * X
        K = WK[h] * X
        V = WV[h] * X
        push!(heads, attention(Q,K,V; mask=false))
    
    end

    print("Size of heads",size(heads))
    X_att = trnsf_Node(:concat, heads, vcat([h.value for h in heads]...))

    return X_att
end

function Transformer_Block(X)

    X_att = multihead_attention(X)

    print("\n Size of X_att ",size(X_att))
    print("\n Size of X ",size(X))
    
    X = X + X_att
    return X
end


function feed_forward(X)
    
    W1 = trnsf_Node(randn(128, size(X, 1)))
    b1 = trnsf_Node(zeros(128))
    W2 = trnsf_Node(randn(size(X, 1), 128))
    b2 = trnsf_Node(zeros(size(X, 1)))
    
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
    
#function last_column(X::trnsf_Node)
#    return X[:, size(X, 2)]
#end

## implementing layer normalization
#= 
function layer_norm(X::trnsf_Node, eps=1e-5)
    mu = mean(X.value; dims=1)
    sigma = mean((X.value .- mu).^2; dims=1)

    ## learnable param
    gamma = trnsf_Node(ones(size(X.value, 1), 1))
    beta = trnsf_Node(zeros(size(X.value, 1), 1))

    X_norm = (X .- mu) ./ sqrt.(sigma .+ eps)
    return gamma .* X_norm .+ beta

end =#

function encoder(X, P)

  
    print("Size of X",size(X))
    print("Size of vector ",typeof(X), "Size of PEM" ,typeof(P))
    print(size(X), size(P.value))
    X = X + P 

    X = Transformer_Block(X)
    
    
    print("Type of X is :", typeof(X))
    ## 
    X_ff = feed_forward(X)
    print("Size of X_ff", size(X_ff))
    X = X + X_ff
    
    print("Size of X after network : ", size(X))
    print("\n Type of X", size(X),"\n")

    return X
end

function decoder_block(X::trnsf_Node, encoder_out::trnsf_Node) 
    ## multihead attention with mask
    ## cross attention
    d_model = size(X, 1)
    num_heads = 4
    
    ## q= decoder K, V = encoder output
    WQ = [trnsf_Node(randn(div(d_model, num_heads), d_model)) for _ in 1:num_heads]
    WK = [trnsf_Node(randn(div(d_model, num_heads), d_model)) for _ in 1:num_heads]
    WV = [trnsf_Node(randn(div(d_model, num_heads), d_model)) for _ in 1:num_heads]
    
    ## for each_head
    heads = []
    for h in 1:num_heads
        Q = WQ[h] * X
        K = WK[h] * X
        V = WV[h] * X
        push!(heads, attention(Q, K, V; mask=true))
    end
    
    X_self = trnsf_Node(:concat, heads, vcat([h.value for h in heads]...))
    
    X = X + X_self

    head_cross = []
    for h in 1:num_heads
        Q = WQ[h] * X_self
        K = WK[h] * encoder_out
        V = WV[h] * encoder_out
        push!(head_cross, attention(Q, K, V; mask=false))
    end

    ## addiding input with X_dec

    ## concatenate 
    X_cross = trnsf_Node(:concat, head_cross, vcat([h.value for h in head_cross]...))
    
    X = X + X_cross

    ## feed forward
    X_ff = feed_forward(X)
    X = X + X_ff
    
    return X
end

function last_column(X::trnsf_Node)
    return X[:, size(X, 2)]
end


function topological_sort(node::trnsf_Node)
    ## define visted block
    ## store the nodes whole set is visit
    visited = Set()
    sorted_order = []

    ## call recursive function dfs
    function DFS(node::trnsf_Node)
        ## check if the node is visited 
        if !(node in visited)
            push!(visited,node)
            
            # iterate over its children
            for args in node.args
                ## call recursive
                DFS(args)
            end
            ## append
            push!(sorted_order, node)
        end
    end
    ## call DFS
    DFS(node)
    return sorted_order
end

function backward(Node::trnsf_Node)
    ## topological order to get the nodes sorted.
    sorted_order = topological_sort(Node)

    ## start with reverse oreder.
    Node.derivative = 1

    for node in reverse(sorted_order)
        print("\n Size :",size(node.value))
        print("\n Op ", node.op)
    

        ## iterate over every node
        if isnothing(node.op)
            continue
        elseif node.op == :+
            x,y = node.args
            print("\n Size of x", size(x))
            print("\n Size of y", size(y))

            ## included for the bias 
            if length(x) == 2 && length(y) == 2  
                for arg in node.args
                    arg.derivative .+= node.derivative
                end
            end

        elseif node.op == :- && length(node.args) == 2

            print("\n The size of node", node.op)
            print("\n The size of node", size(node.value))

            x,y = node.args
            x.derivative .+= node.derivative
            y.derivative .+= -node.derivative

        elseif node.op == :- && length(node.args) == 1
            x = node.args[1]
            x.derivative = node.derivative * -1

        elseif node.op == :*
            x,y = node.args
            print("\n The size of x in * op", size(x.value))
            print("\n The size of y in * op", size(y.value))
            print("\n The size of nd in * op", size(node.derivative))
            
            if isa(x.value, AbstractArray) && isa(y.value, AbstractArray)
                x.derivative += node.derivative * y.value'
                y.derivative += x.value' * node.derivative
            
            elseif isa(x.value, Number) && isa(y.value, AbstractArray)
                print("\n The x.valye is ",x.value)
                ## this is removed because we were implement W =  (rand) 
                #x.derivative += node.derivative * y.value'
                y.derivative +=  node.derivative
            
            end

        elseif node.op == :/
            x,y = node.args
            println("Backward getindex: size(x.dev)=", size(x.derivative),", size(incoming)=", size(node.derivative))
            ## case handled
            if isa(x.value, AbstractArray) && isa(y.value, Number)
            
                x.derivative .+= node.derivative .* y.value
                y.derivative .+= x.value' * node.derivative
            end

        elseif node.op == :log
            x = node.args[1]
            x.derivative = node.derivative .* 1/x.value
        
        elseif node.op == :softmax
            x = node.args[1]
            s = node.value
            
            x.derivative = node.value .* (node.derivative .- sum(node.value .* node.derivative; dims=2))

        elseif node.op == :relu
            x = node.args[1]
            grad_relu = (x.value .> 0)
            x.derivative = grad_relu .* node.derivative
        else
            error(" The opertion `node.op` is not implemented")
        end

    end
end


function topological_sort(node::trnsf_Node)
    ## define visted block
    ## store the nodes whole set is visit
    visited = Set()
    sorted_order = []

    ## call recursive function dfs
    function DFS(node::trnsf_Node)
        ## check if the node is visited 
        if !(node in visited)
            push!(visited,node)
            
            # iterate over its children
            for args in node.args
                ## call recursive
                DFS(args)
            end
            ## append
            push!(sorted_order, node)
        end
    end
    ## call DFS
    DFS(node)
    return sorted_order
end

function backward(Node::trnsf_Node)
    ## topological order to get the nodes sorted.
    sorted_order = topological_sort(Node)

    ## start with reverse oreder.
    node.derivative = 1

    for node in reverse(sorted_order)
        print("\n Size :",size(node.value))
        print("\n Op ", node.op)
    

        ## iterate over every node
        if isnothing(node.op)a
            continue
        elseif node.op == :+
            x,y = node.args
            print("\n Size of x", size(x))
            print("\n Size of y", size(y))

            ## included for the bias 
            if length(x) == 2 && length(y) == 2  
                for arg in node.args
                    arg.derivative .+= node.derivative
                end
            end

        elseif node.op == :- && length(node.args) == 2

            print("\n The size of node", node.op)
            print("\n The size of node", size(node.value))

            x,y = node.args
            x.derivative .+= node.derivative
            y.derivative .+= -node.derivative

        elseif node.op == :- && length(node.args) == 1
            x = node.args[1]
            x.derivative = node.derivative * -1

        elseif node.op == :*
            x,y = node.args
            print("\n The size of x in * op", size(x.value))
            print("\n The size of y in * op", size(y.value))
            print("\n The size of nd in * op", size(node.derivative))
            
            if isa(x.value, AbstractArray) && isa(y.value, AbstractArray)
                x.derivative += node.derivative * y.value'
                y.derivative += x.value' * node.derivative
            
            elseif isa(x.value, Number) && isa(y.value, AbstractArray)
                print("\n The x.valye is ",x.value)
                ## this is removed because we were implement W =  (rand) 
                #x.derivative += node.derivative * y.value'
                y.derivative +=  node.derivative
            
            end

        elseif node.op == :/
            x,y = node.args
            println("Backward getindex: size(x.dev)=", size(x.derivative),
        ", size(incoming)=", size(node.derivative))
            ## case handled
            if isa(x.value, AbstractArray) && isa(y.value, Number)
            
                x.derivative .+= node.derivative .* y.value
                y.derivative .+= x.value' * node.derivative
            end

        elseif node.op == :log
            x = node.args[1]
            x.derivative = node.derivative .* 1/x.value
        
        elseif node.op == :getindex
            inds = node.meta[:inds]
            print("\n Inside gt ",inds)
            x = node.args[1]

            ##intialize parents 
            ## two cases
            if isa(inds, Tuple)
                row_inds, col_inds = inds
                ## the comming derivaitve should only be updates for the selected colums
                ## normally the last one 
                x.derivative[row_inds, col_inds] .+= node.derivative
            else
                idx = inds
                x.derivative[idx] += node.derivative
            end

            print("\n Size of x.dev" , size(x.derivative))
            println("Backward getindex: size(x.dev)=", size(x.derivative),
        ", size(incoming)=", size(node.derivative))
      
        elseif node.op == :softmax
            x = node.args[1]
            s = node.value
            
            x.derivative = node.value .* (node.derivative .- sum(node.value .* node.derivative; dims=2))

        elseif node.op == :relu
            x = node.args[1]
            grad_relu = (x.value .> 0)
            x.derivative = grad_relu .* node.derivative
        elseif node.op == :neg
            x = node.args[1]
            x.derivative += node.derivaitve * (-1/ seq_len)
        else
            error(" The opertion `node.op` is not implemented")
        end
    end
end


# 10,1 
first_seq = X[1]
print("size of first seq",size(first_seq))

first_seq = embed(first_seq)

print("\n size of X is :", size(first_seq))

P = trnsf_Node(randn(d_model, block_size))
print("\n tye of P :" ,typeof(P))

encoder_out = encoder(first_seq, P)
println("size of the encoder_out is : ",size(encoder_out))
println("type of the encoder_out is : ",typeof(encoder_out))


X_decoded = decoder_block(first_seq, encoder_out)
println("type of the X_decoded is : ",typeof(X_decoded))
println("size of the X_decoded is : ",size(X_decoded))

W_out =  trnsf_Node(randn(vocab_size, d_model))

## Projecting into multi d space.


logits = W_out * X_decoded

prob = softmax_node(logits)

target_seq = Y[1]

seq_len = size(X_decoded, 2)

sum_logs = trnsf_Node(0.0)

for t in 1:seq_len
    tgt = target_seq[t]

    p_t = prob[tgt, t]

    ## create a log node
    log_p = trnsf_Node(:log, [p_t], log.(p_t.value))

    ## accumulate
    sum_logs = sum_logs + log_p

end

loss_node = trnsf_Node(:neg, [sum_logs], -sum_logs.value / seq_len)

println("Loss (mean NLL): ", loss_node.value)

# Backpropagate through the whole expression graph:
backward(loss_node)   # your backward function expects a trnsf_Node