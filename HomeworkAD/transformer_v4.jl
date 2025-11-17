using HTTP
using Unicode
using Flux
using Statistics


# 1. Download dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
resp = HTTP.get(url)
text = String(resp.body)

# 2. Preprocess: lowercase, keep a reasonable set of chars
text = lowercase(text)
# Keep letters, numbers, punctuation, spaces and newlines
text = filter(c -> isletter(c) || isspace(c) || isdigit(c) || c in ['.',',','!','?',';',':','\'','"','-'], text)

# 3. Work at character level
chars = collect(text)                
unique_chars = unique(chars)      

# 4. Display info
println("Total chars: ", length(chars))
println("Unique chars: ", length(unique_chars))
println("Unique chars themselves: ", unique_chars)


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
    meta::Dict{Symbol, Any} 

end

## Fixed constructors

## Constructor for operations (arrays)
trnsf_Node(op, args, value::AbstractArray) = trnsf_Node(op, args, value, zeros(size(value)), Dict(:shape => size(value)))

## Constructor for operations (scalars)
trnsf_Node(op, args, value::Number) = trnsf_Node(op, args, value, 0.0, Dict(:shape => (1,)))

## Constructor for leaf nodes (arrays)
trnsf_Node(x::AbstractArray) = trnsf_Node(nothing, trnsf_Node[], x, zeros(size(x)), Dict(:shape => size(x)))

## Constructor for leaf nodes (scalars) - THIS IS THE KEY FIX
trnsf_Node(x::Number) = trnsf_Node(nothing, trnsf_Node[], x, 0.0, Dict(:shape => (1,)))


Base.:*(x::AbstractArray, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x * y)
Base.:*(x::Number, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x .* y )
Base.:*(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:*, [x, y], x.value * y.value)

Base.:*(x::Number, y::AbstractArray) = trnsf_Node(:*, [trnsf_Node(x), trnsf_Node(y)], x .* y )


Base.:+(x::AbstractArray, y::AbstractArray) = trnsf_Node(:+, [trnsf_Node(x), trnsf_Node(y)], x + y)
Base.:+(x::AbstractArray, y::trnsf_Node) = trnsf_Node(:+, [trnsf_Node(x), y], x + y.value)

Base.:+(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:+, [x, y], x.value .+ y.value)

Base.:+(x::trnsf_Node, y::AbstractVector) = trnsf_Node(:+, [x, trnsf_Node(y)], x.value .+ y)



Base.:-(x::trnsf_Node, y::AbstractArray) = trnsf_Node(:-, [x, trnsf_Node(y)], x.value .- y)
Base.:-(x::trnsf_Node) = trnsf_Node(:neg, [x], -x.value)

## Overloading the adjoint 
Base.:adjoint(x::trnsf_Node) = trnsf_Node(:adjoint, [x], x.value')
Base.sqrt(x::trnsf_Node)  = trnsf_Node(:sqrt, [x], sqrt.(x.value))
#Base.maximum(n::trnsf_Node, dims::Int) = maximum(n.value, dims= dims) 


Base.:/(x::trnsf_Node, y::Number) = trnsf_Node(:/ , [x, trnsf_Node(y)], x.value ./ y) 
Base.:/(x::trnsf_Node, y::trnsf_Node) = trnsf_Node(:/ , [x, y], x.value ./ y.value) 

relu(x) = max(0, x)

relu(x::trnsf_Node) = trnsf_Node(:relu, [x], relu.(x.value))
    
#Base.:transpose(x::trnsf_Node) = trnsf_Node(:transpose, [x], x.value')

## Overloading the base operator
Base.size(n::trnsf_Node) = size(n.value)
Base.size(n::trnsf_Node, dim::Int) = size(n.value, dim)
Base.length(n::trnsf_Node) = length(n.value)

#Base.getindex(n::trnsf_Node, inds...) = n.value[inds...]

Base.broadcastable(x::trnsf_Node) = Broadcast.broadcastable(x.value)
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


function softmax_node(scores::trnsf_Node; eps=1e-12)
    max_val = maximum(scores.value, dims=1)
    exp_s = exp.(scores.value .- max_val)
    sm = exp_s ./ (sum(exp_s, dims=1) .+ eps)
    return trnsf_Node(:softmax, [scores], sm)
end

##--------------------------------

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
function backward(node::trnsf_Node)
    sorted_order = topological_sort(node)
    node.derivative = 1

    for node in reverse(sorted_order)
        if isnothing(node.op)
            continue
            
        elseif node.op == :+
            for arg in node.args
                if size(arg.derivative) == size(node.derivative)
                    if isa(arg.derivative, AbstractArray)
                        arg.derivative .+= node.derivative
                    else
                        arg.derivative += node.derivative
                    end
                else
                    # Broadcasting case - sum over columns for bias
                    arg.derivative .+= sum(node.derivative, dims=2)
                end
            end

        elseif node.op == :*
            x, y = node.args
            x.derivative .+= node.derivative * y.value'
            y.derivative .+= x.value' * node.derivative
           

        elseif node.op == :/
            x, y = node.args
            if isa(x.derivative, AbstractArray)
                x.derivative .+= node.derivative ./ y.value
            else
                x.derivative += node.derivative / y.value
            end
            

        elseif node.op == :log
            x = node.args[1]
            if isa(x.derivative, AbstractArray)
                x.derivative .+= node.derivative ./ x.value
            else
                x.derivative += node.derivative / x.value
            end
        
        elseif node.op == :getindex
            inds = node.meta[:inds]
            x = node.args[1]
            row_inds, col_inds = inds
            if isa(node.derivative, AbstractArray)
                x.derivative[row_inds, col_inds] .+= node.derivative
            else
                # Scalar case if node.derivative is number while I want to store it in array
                x.derivative[row_inds, col_inds] += node.derivative
            end
            
        elseif node.op == :adjoint
            x = node.args[1]
            x.derivative .+= node.derivative'

        elseif node.op == :softmax
            x = node.args[1]
            x.derivative .+= node.value .* (node.derivative .- sum(node.value .* node.derivative; dims=1))
        
        elseif node.op == :relu
            x = node.args[1]
            x.derivative .+= (x.value .> 0) .* node.derivative            
            
        elseif node.op == :neg
            x = node.args[1]
            if isa(x.derivative, AbstractArray)
                x.derivative .+= -node.derivative
            else
                x.derivative += -node.derivative
            end
            
        elseif node.op == :concat
            heads = node.args
            start_row = 1
            for head in heads
                ## get the sum of row as they concatenated like this 
                n_rows = size(head.value, 1)
                head.derivative .+= node.derivative[start_row:start_row + n_rows - 1, :]
                start_row += n_rows
            end
        
        elseif node.op == :masked_scores
            x = node.args[1]
            x.derivative .+= node.derivative

        elseif node.op == :embed
            W = node.args[1]
            seq = node.meta[:seq]
            #dnode = node.derivative

            for (i, idx) in enumerate(seq)
                #if isa(dnode, AbstractArray)
                    W.derivative[:, idx] .+= node.derivative[:, i]
                #else   
                #     W.derivative[:, idx] .+= fill(dnode, size(W.derivative, 1))
                #end
            end
        else
            error("The operation $(node.op) is not implemented")
        end
    end
end

char2id = Dict{Char, Int}()
id2char = Dict{Int, Char}()

for (i, c) in enumerate(unique_chars)
    char2id[c] = i
    id2char[i] = c
end

encoded_text = [char2id[c] for c in chars]  # Vector{Int}




block_size = 32
X = []
Y = []

# converting data into blocks
for i in 1:(length(encoded_text) - block_size)
    push!(X, encoded_text[i: i+block_size-1])
    push!(Y, encoded_text[i+1: i+block_size])
end

println("Length of X: ", length(X))

## define an embedding dims 
using LinearAlgebra
using Random

vocab_size = length(unique_chars)
d_model = 64              # can stay the same for now
   

num_heads = 4
d_head = div(d_model, num_heads)

xavier(dims...) = randn(dims...) * sqrt(2.0/ sum(dims))

WQ_enc = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]
WK_enc = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]
WV_enc = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]

# Decoder attention weights  
WQ_dec = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]
WK_dec = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]
WV_dec = [trnsf_Node(xavier(d_head, d_model)) for _ in 1:num_heads]

# Feed-forward layers
W1_ff = trnsf_Node(xavier(128, d_model))
b1_ff = trnsf_Node(zeros(128))
W2_ff = trnsf_Node(xavier(d_model, 128))
b2_ff = trnsf_Node(zeros(d_model))

W_embed = trnsf_Node(xavier(d_model, vocab_size))
P = trnsf_Node(randn(d_model, block_size) * 0.01)      
W_out = trnsf_Node(xavier(vocab_size, d_model))

# ===== MODIFIED FUNCTIONS USING GLOBAL PARAMETERS =====
function embed(seq::Vector{Int})
    X_emb = hcat([W_embed.value[:, id] for id in seq]...)
    node = trnsf_Node(:embed, [W_embed], X_emb)
    ## sequence get stored as a list
    node.meta[:seq] = seq  # Store the sequence for backward pass
    return node
end

function attention(Q::trnsf_Node, K::trnsf_Node, V::trnsf_Node; mask=false)
    dk = size(K, 1)
    scores = (Q' * K) / sqrt(dk)
    
    if mask 
        seq_len = size(scores, 2)
        mask_matrix = triu(ones(seq_len, seq_len), 1) .* -1e9
        scores = trnsf_Node(:masked_scores, [scores], scores.value .+ mask_matrix)
        #scores = scores.value .+ mask_matrix
    end
    
    scores = softmax_node(scores)
    return V * scores 
end

function multihead_attention(X, WQ, WK, WV, num_heads=4)
    d_model = size(X, 1)
 
    
    heads = []
    for h in 1:num_heads
        Q = WQ[h] * X
        K = WK[h] * X
        V = WV[h] * X
        head = attention(Q, K, V)
        
        push!(heads, attention(Q, K, V; mask=false))
    end
    
    X_att = trnsf_Node(:concat, heads, vcat([h.value for h in heads]...))
    return X_att
end

function Transformer_Block(X)
    X_att = multihead_attention(X, WQ_enc, WK_enc, WV_enc)
    X = X + X_att
    return X
end

function feed_forward(X)
    # Layer # 1
    l1 = W1_ff * X 
    l1 = l1 + b1_ff  
    
    ## Activation
    l1 = relu(l1)
    
    ## Layer # 2
    l2 = W2_ff * l1 
    y_pred = l2 + b2_ff
    
    return y_pred
end

function encoder(X, P)
    X = X + P 
    X = Transformer_Block(X)
    
    X_ff = feed_forward(X)
    X = X + X_ff
    
    return X
end

function decoder_block(X::trnsf_Node, encoder_out::trnsf_Node) 
    d_model = size(X, 1)
    num_heads = 4
    
    # Self-attention with masking
    heads = []
    for h in 1:num_heads
        Q = WQ_dec[h] * X
        K = WK_dec[h] * X
        V = WV_dec[h] * X
        ## output of trnsfnode 
        ## each head has some computation graph 

        push!(heads, attention(Q, K, V; mask=true))
    end
    ## connecting X_att with previous 
    X_self = trnsf_Node(:concat, heads, vcat([h.value for h in heads]...))
    
    X = X + X_self

    # Cross-attention (no masking)
    head_cross = []
    for h in 1:num_heads
        Q = WQ_dec[h] * X_self
        K = WK_dec[h] * encoder_out
        V = WV_dec[h] * encoder_out
        push!(head_cross, attention(Q, K, V; mask=false))
    end

    X_cross = trnsf_Node(:concat, head_cross, vcat([h.value for h in head_cross]...))

    X = X + X_cross

    # Feed forward
    X_ff = feed_forward(X)
    X = X + X_ff
    
    return X
end


function collect_params(node::trnsf_Node, params=Set{trnsf_Node}())
    if node.op === nothing  # leaf node
        if isa(node.value, AbstractArray)
            push!(params, node)
        end
    end
    for arg in node.args
        collect_params(arg, params)
    end
    return params
end

function reset_grads!(params)
    for p in params
        if isa(p.derivative, AbstractArray)
            p.derivative .= 0.0
        elseif isa(p.derivative, Number)
            p.derivative = 0.0
        end
    end
end

# ===== TRAINING LOOP =====

function train!(n_epochs::Int; η_start=0.005, η_end=1e-5)
    global losses
    losses = Float64[]
    
    total_steps = n_epochs * length(X)
    step = 0

    println("\n=== Starting Training ===\n")

    for epoch in 1:n_epochs
        idxs = shuffle(1:length(X))   # random order each epoch

        for idx in idxs
            step += 1

            # Learning rate schedule (linear decay)
            progress = step / total_steps
            η = η_start * (1 - progress) + η_end * progress

            seq    = X[idx]
            target = Y[idx]

            # ----- Forward pass -----
            seq_embedded = embed(seq)

            enc_out = encoder(seq_embedded, P)
            dec_out = decoder_block(seq_embedded, enc_out)

            logits = W_out * dec_out
            probs  = softmax_node(logits)

            seq_len = size(dec_out, 2)

            sum_logs = trnsf_Node(0.0)
            for t in 1:seq_len
                tgt = target[t]
                p_t = probs[tgt, t]
                log_p = trnsf_Node(:log, [p_t], log(p_t.value + 1e-9))
                sum_logs = sum_logs + log_p
            end

            mean_log = sum_logs / trnsf_Node(seq_len)
            loss = trnsf_Node(:neg, [mean_log], -mean_log.value)

            # ----- Backward -----
            backward(loss)
            push!(losses, loss.value)

            # Collect and update parameters
            params = collect_params(loss)

            for p in params
                if isa(p.value, AbstractArray) && isa(p.derivative, AbstractArray)
                    p.value .-= η .* p.derivative
                elseif isa(p.value, Number)
                    p.value -= η * p.derivative
                end
            end

            reset_grads!(params)

            # Logging
            if step % 1000 == 0
                avg_loss = mean(losses[max(1, end-999):end])
                println("step $step | epoch $epoch | loss: $(round(loss.value, digits=3)) | avg1000: $(round(avg_loss, digits=3)) | LR: $(round(η, digits=5))")
            end
        end
    end

    println("\n=== Training Complete ===")
end