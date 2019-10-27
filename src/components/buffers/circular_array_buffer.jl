export CircularArrayBuffer, capacity, isfull, consecutive_view

"""
    CircularArrayBuffer{T}(d::Int...) -> CircularArrayBuffer{T, N}

Using a `N` dimension Array to simulate a circular buffer of `N-1` dimensional elements.

# Examples

```julia-repl
julia> b = CircularArrayBuffer{Float64}(2, 2)
2×0 CircularArrayBuffer{Float64,2}

julia> push!(b, rand(2))
2×1 CircularArrayBuffer{Float64,2}:
 0.4867189515370427
 0.6439720991119842

julia> push!(b, rand(2))
2×2 CircularArrayBuffer{Float64,2}:
 0.486719  0.274857
 0.643972  0.462211

julia> push!(b, rand(2))
2×2 CircularArrayBuffer{Float64,2}:
 0.274857  0.763105
 0.462211  0.194233
```
"""
mutable struct CircularArrayBuffer{T,N} <: AbstractArray{T,N}
    buffer::Array{T,N}
    first::Int
    length::Int
    step_size::Int

    function CircularArrayBuffer{T}(d::Integer...) where {T}
        N = length(d)
        N > 0 || throw(ArgumentError("dimension must be greater than 0"))
        new{T,N}(Array{T}(undef, d...), 1, 0, N == 1 ? 1 : *(d[1:end-1]...))
    end
end

Base.size(cb::CircularArrayBuffer{T,N}) where {T,N} = (size(cb.buffer)[1:N-1]..., cb.length)
Base.IndexStyle(::CircularArrayBuffer) = IndexLinear()
Base.getindex(cb::CircularArrayBuffer{T, N}, i::Int) where {T, N} = getindex(cb.buffer, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T, N}, v, i::Int) where {T, N} = setindex!(cb.buffer, v, _buffer_index(cb, i))

function _buffer_index(cb::CircularArrayBuffer, i::Int)
    ind = (cb.first - 1) * cb.step_size + i
    if ind > length(cb.buffer)
        ind - length(cb.buffer)
    else
        ind
    end
end

capacity(cb::CircularArrayBuffer{T, N}) where {T, N} = size(cb.buffer, N)
isfull(cb::CircularArrayBuffer) = cb.length == capacity(cb)
Base.isempty(cb::CircularArrayBuffer) = cb.length == 0

function Base.empty!(cb::CircularArrayBuffer)
    cb.first = 1
    cb.length = 0
    cb
end

"""
    push!(cb::CircularArrayBuffer, data::AbstractArray)

Add `data` to the back and overwrite front if full.
"""
function Base.push!(cb::CircularArrayBuffer{T,N}, data) where {T,N}
    length(data) == cb.step_size || throw(ArgumentError("length of , $(cb.step_size) != $(length(data))"))
    if cb.length == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.length += 1
    end
    selectdim(cb, N, cb.length) .= data
    cb
end

function consecutive_view(b::CircularArrayBuffer{T,N}, inds, n) where {T,N}
    expanded_inds = collect(Iterators.flatten(x:x+n-1 for x in inds))
    reshape(selectdim(b, N, expanded_inds), size(b.buffer)[1:N-1]..., n, length(inds))
end

function consecutive_view(b::CircularArrayBuffer{T,N}, inds, n, n_frames) where {T,N}
    expanded_inds = collect(Iterators.flatten((i-n_frames+1):i for i in inds for j in i:i+n-1))
    reshape(selectdim(b, N, expanded_inds), size(b.buffer)[1:N-1]..., n_frames, n, length(inds))
end

consecutive_view(b::CircularArrayBuffer{T,N}, inds, n, n_frames::Nothing) where {T,N} = consecutive_view(b, inds, n)