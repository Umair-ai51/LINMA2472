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
    meta::Dict{Symbol, Any} 

end


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
Base.:-(x::trnsf_Node) = trnsf_Node(:neg, [x], -x.value)

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

 
##-------------
function softmax_node(scores::trnsf_Node; eps=1e-9)
    score_stable = scores.value .- maximum(scores.value, dims=2)
    exp_scores = exp.(clamp.(score_stable, -50, 50))  # clip extreme values
    softmax_Scores = exp_scores ./ (sum(exp_scores, dims=2) .+ eps)
    softmax_Scores = clamp.(softmax_Scores, eps, 1.0 - eps)  # prevent zeros
    return trnsf_Node(:softmax, [scores], softmax_Scores)
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
    ## topological order to get the nodes sorted.
    sorted_order = topological_sort(node)

    ## start with reverse oreder.
    node.derivative = 1

    for node in reverse(sorted_order)
        ## iterate over every node
        if isnothing(node.op)
            continue
        elseif node.op == :+
            # SIMPLE: For addition, handle broadcasting by summing over columns
            for arg in node.args
                if size(arg.derivative) == size(node.derivative)
                    # Same shape - direct addition
                    arg.derivative .+= node.derivative
                else
                    # Different shapes - sum over columns (most common case)
                    arg.derivative .+= sum(node.derivative, dims=2)
                end
            end


         elseif node.op == :- && length(node.args) == 2
            x, y = node.args
            # Handle subtraction with same logic
            if size(x.derivative) == size(node.derivative)
                x.derivative .+= node.derivative
            else
                x.derivative .+= sum(node.derivative, dims=2)
            end
            
            if size(y.derivative) == size(node.derivative)
                y.derivative .+= -node.derivative
            else
                y.derivative .+= -sum(node.derivative, dims=2)
            end

        elseif node.op == :*
            x,y = node.args
        
            if isa(x.value, AbstractArray) && isa(y.value, AbstractArray)
                x.derivative .+= node.derivative * y.value'
                y.derivative .+= x.value' * node.derivative
            
            elseif isa(x.value, Number) && isa(y.value, AbstractArray)
                x.deriavtive .+= node.derivative .* y.value
                y.derivative .+= x.value .* node.derivative  
            end

        elseif node.op == :/
            x,y = node.args
            if isa(x.value, AbstractArray) && isa(y.value, Number)
            
                x.derivative .+= node.derivative ./ y.value

                if isa(y.derivative, AbstractArray)
                    #print("\n Scaler y.derivaitve")
                    y.derivative .+= sum(-x.value .* node.derivative) / (y.value^2)
                else
                    y.derivative += sum(-x.value .* node.derivative) / (y.value^2)
                end
            end

        elseif node.op == :log
            x = node.args[1]
            x.derivative = node.derivative .* 1/x.value
        
        elseif node.op == :getindex
            inds = node.meta[:inds]
            #print("\n Inside gt ", inds)
            x = node.args[1]

                ##intialize parents 
                ## two cases
            if isa(inds, Tuple)
                row_inds, col_inds = inds
                
        
                # Check if node.derivative is a scalar or array
                if isa(node.derivative, AbstractArray)
                    x.derivative[row_inds, col_inds] .+= node.derivative
                else
                    # node.derivative is a scalar (e.g., from prob[tgt, t])
                    x.derivative[row_inds, col_inds] += node.derivative
                end
            else
                idx = inds
                if isa(node.derivative, AbstractArray)
                    x.derivative[idx] .+= node.derivative
                else    
                    x.derivative[idx] += node.derivative
                end
        end
        
        elseif node.op == :adjoint
            x = node.args[1]
            # Gradient of transpose: just transpose the incoming gradient back
            x.derivative .+= node.derivative'

        elseif node.op == :softmax
            x = node.args[1]
            s = node.value
            
            x.derivative .+= node.value .* (node.derivative .- sum(node.value .* node.derivative; dims=2))
        
        
        elseif node.op == :relu
            x = node.args[1]
            grad_relu = (x.value .> 0)
            x.derivative += grad_relu .* node.derivative
            
        elseif node.op == :neg
            x = node.args[1]
            # Gradient of -x is just -1
            if isa(x.derivative, AbstractArray)
                x.derivative .+= -node.derivative
            else
                x.derivative += -node.derivative
            end
        elseif node.op == :concat
            heads = node.args
            d_model_per_head = size(heads[1].value, 1)  # number of rows per head
    
            start_row = 1
            for head in heads
                n_rows = size(head.value, 1)
        # Extract the gradient slice for this head
                head.derivative .+= node.derivative[start_row:start_row + n_rows - 1, :]
                start_row += n_rows
            end
        
        elseif node.op == :masked_scores
            x = node.args[1]
            x.derivative .+= node.derivative

        elseif node.op == :embed
            W = node.args[1]            # W_embed node
            seq = node.meta[:seq]       # stored sequence indices
            dnode = node.derivative      # already numeric

            for (i, idx) in enumerate(seq)
                if isa(dnode, AbstractArray)
                    W.derivative[:, idx] .+= dnode[:, i]
                else
                    W.derivative[:, idx] .+= fill(dnode, size(W.derivative, 1))
                end
            end
        
        else
            error(" The opertion `node.op` is not implemented")
        end
    end
end
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

println("Length of X: ", length(X))

## define an embedding dims 
using LinearAlgebra
using Random

vocab_size = length(unique_words)
d_model = 64    

# ===== GLOBAL PARAMETER INITIALIZATION =====
# Embedding
W_embed = trnsf_Node(randn(d_model, vocab_size))

# Positional encoding (fixed)
P = trnsf_Node(randn(d_model, block_size))

# Output projection
W_out = trnsf_Node(randn(vocab_size, d_model))

# Multihead attention parameters
num_heads = 4
d_head = div(d_model, num_heads)

# Encoder attention weights
WQ_enc = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
WK_enc = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
WV_enc = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]

# Decoder attention weights  
WQ_dec = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
WK_dec = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]
WV_dec = [trnsf_Node(randn(d_head, d_model)) for _ in 1:num_heads]

# Feed-forward layers
W1_ff = trnsf_Node(randn(128, d_model))
b1_ff = trnsf_Node(zeros(128))
W2_ff = trnsf_Node(randn(d_model, 128))
b2_ff = trnsf_Node(zeros(d_model))

# ===== MODIFIED FUNCTIONS USING GLOBAL PARAMETERS =====
function embed(seq::Vector{Int})
    X_emb = hcat([W_embed.value[:, id] for id in seq]...)
    node = trnsf_Node(:embed, [W_embed], X_emb)
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
    end
    
    scores = softmax_node(scores)
    return V * scores 
end

function multihead_attention(X, WQ, WK, WV, num_heads=4)
    d_model = size(X, 1)
    d_head = div(d_model, num_heads)
    
    heads = []
    for h in 1:num_heads
        Q = WQ[h] * X
        K = WK[h] * X
        V = WV[h] * X
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
    println("Size of X: ", size(X))
    println("Size of P: ", size(P.value))
    X = X + P 
    X = Transformer_Block(X)
    
    println("Type of X after Transformer: ", typeof(X))
    X_ff = feed_forward(X)
    println("Size of X_ff: ", size(X_ff))
    X = X + X_ff
    
    println("Size of X after network: ", size(X))
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
        push!(heads, attention(Q, K, V; mask=true))
    end
    
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
println("\n=== Starting Training ===\n")

# Training hyperparameters
η = 1e-3
num_steps = min(10, length(X))  # Ensure we don't exceed available data

for step in 1:num_steps
    println("\n--- Training Step $step ---")
    
    # Get training example
    seq = X[step]
    target = Y[step]
    
    # Embed the sequence (uses global W_embed)
    seq_embedded = embed(seq)
    
    # Forward pass (uses all global parameters)
    enc_out = encoder(seq_embedded, P)
    dec_out = decoder_block(seq_embedded, enc_out)
    
    # Output projection (uses global W_out)
    logits = W_out * dec_out
    probs = softmax_node(logits)
    
    # Compute mean negative log-likelihood loss
    seq_len = size(dec_out, 2)
    sum_logs = trnsf_Node(0.0)

    for t in 1:seq_len
        ## get the target index
        tgt = target[t]
        
        ## get the prob index 
        p_t = probs[tgt, t]
        log_p = trnsf_Node(:log, [p_t], log(p_t.value))
        sum_logs = sum_logs + log_p
    end
    mean_log = sum_logs / seq_len
    loss = trnsf_Node(:neg, [mean_log], -mean_log.value)
    
    println("Loss: ", loss.value)
    
    # Backward pass
    backward(loss)
    
    # Collect all trainable parameters automatically
    params = collect_params(loss)
    println("Updating ", length(params), " trainable parameters.")
    
    # Gradient descent update for all parameters
    for p in params
        if isa(p.value, AbstractArray) && isa(p.derivative, AbstractArray)
            p.value .-= η .* p.derivative
        elseif isa(p.value, Number)
            p.value -= η * p.derivative
        end
    end
    
    # Reset all gradients
    reset_grads!(params)
    
    println("Step $step complete")
end

println("\n=== Training Complete ===")