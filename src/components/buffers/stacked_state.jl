export StackedState

using Flux:@forward

"""
    StackedState(; state_size, state_eltype=Float32, n_frames=4)

Use a [`CircularArrayBuffer`](@ref) to store stacked states.

# Example
"""
struct StackedState{T, N} <: AbstractArray{T, N}
    states::CircularArrayBuffer{T, N}
end

function StackedState(; state_size, state_eltype=Float32, n_frames=4)
    states = CircularArrayBuffer{state_eltype}(state_size, n_frames)
    for _ in 1:n_frames
        push!(states, zeros(state_eltype, state_size))
    end
    StackedState(states)
end

@forward StackedState.states Base.push!, Base.getindex, Base.setindex!

Base.push!(b::CircularArrayBuffer, s::StackedState) = push!(b, select_last_frame(s))