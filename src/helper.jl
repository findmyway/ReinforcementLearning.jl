@inline function maximumbelowInf(values)
    m = -Inf64
    for v in values
        if v < Inf64 && v > m
            m = v
        end
    end
    if m == -Inf64
        Inf64
    else
        m
    end
end

@inline getvalue(params, state::Int) = params[:, state]
@inline getvalue(params::Vector, state::Int) = params[state]
@inline getvalue(params, action::Int, state::Int) = params[action, state]
@inline getvalue(params, state::AbstractArray) = params * state
@inline getvalue(params::Vector, state::Vector) = dot(params, state)
@inline getvalue(params, action::Int, state::AbstractArray) = 
    dot(view(params, action, :), state)

function findallmax(arr)
    max_idxs = Vector{Int}()
    max_val = typemin(eltype(arr))
    for i in eachindex(arr)
        if arr[i] > max_val
            max_val = arr[i]
            empty!(max_idxs)
            push!(max_idxs, i)
        elseif arr[i] == max_val
            push!(max_idxs, i)
        end
    end
    (max_idxs=max_idxs, max_val=max_val)
end