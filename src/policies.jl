export SoftmaxPolicy, EpsilonGreedyPolicy, ForcedPolicy
import StatsBase:sample

abstract type AbstractPolicy end
function sample end
function getprob end

sample(p::AbstractPolicy, values, actions) = actions[sample(p, values)]

##############################
## SoftmaxPolicy
##############################

"""
Returns a SoftmaxPolicy with default β = 1.

    mutable struct SoftmaxPolicy <: AbstractPolicy
        β::Float64
    end
"""
struct SoftmaxPolicy <: AbstractPolicy
    β::Float64
    SoftmaxPolicy() = new(1.0)
end

"""
Return a softmax weighted random sample's index.
"""
function sample(p::SoftmaxPolicy, values)
    if p.β == Inf
        rand(findallmax(vs).max_idxs)
    else
        max_idxs, max_val = findallmax(values)
        if max_val == typemax(eltype(values))
            rand(max_idxs)
        else
            wsample(exp.(p.β .* values))
        end
    end
end

function getprob(policy::SoftmaxPolicy, values)
    max_idxs, max_val = findallmax(values)
    if policy.β == Inf || max_val == typemax(eltype(values))
        p = zero(values)
        for i in max_idxs
            p[i] = 1/length(max_idxs)
        end
        p
    else
        expvalues = exp.(policy.β .* (values .- maximum(values)))
        expvalues/sum(expvalues)
    end
end


##############################
## EpsilonGreedyPolicy
##############################
"""
    struct EpsilonGreedyPolicy
        ϵ::Float64

Chooses the action with the highest value with probability `1 - ϵ` and selects 
an action uniformly random with probability `ϵ`.
"""
struct EpsilonGreedyPolicy <: AbstractPolicy
    ϵ::Float64
end

function sample(p::EpsilonGreedyPolicy, values)
    rand() < p.ϵ ? rand(1:length(values)) : rand(findallmax(values).max_idxs)
end

function getprob(policy::EpsilonGreedyPolicy, values)
    p = ones(length(values))/length(values) * policy.ϵ
    max_idxs, max_val = findallmax(values)
    p_offset = (1. - policy.ϵ)/length(max_idxs)
    for i in max_idxs
        p[i] += p_offset
    end
    p
end

struct NMarkovPolicy{N, Tpol, Tbuf}
    policy::Tpol
    buffer::Tbuf
end
function NMarkovPolicy(N, pol::Tpol, buf::Tbuf) where {Tpol, Tbuf}
    NMarkovPolicy{N, Tpol, Tbuf}(pol, buf)
end
function (p::NMarkovPolicy{N, Tpol, Tbuf})(s) where {N, Tpol, Tbuf}
    push!(p.buffer, s)
    p.policy(nmarkovgetindex(p.buffer, N, N))
end
function defaultnmarkovpolicy(learner, buffer, π)
    if learner.nmarkov == 1
        π
    else
        a = buffer.states.data
        data = getindex(a, map(x -> 1:x, size(a)[1:end-1])..., 1:learner.nmarkov)
        NMarkovPolicy(learner.nmarkov, 
                      π, 
                      ArrayCircularBuffer(data, learner.nmarkov, 0, 0, false))
    end
end

##############################
## ForcedPolicy
##############################
mutable struct ForcedPolicy 
    n::Int
    cur::Int
    ForcedPolicy(n::Int) = new(n, 1)
end

function sample(p::ForcedPolicy, values) 
    # length(values) == p.n || error("lenght of $values doesn't match with $p")
    cur, nxt = p.cur, p.cur + 1
    p.cur = nxt > p.n ? 1 : nxt
    cur
end

function getprob(p::ForcedPolicy, values) 
    # length(values) == p.n || error("lenght of $values doesn't match with $p")
    fill(1/p.n, size(values))
end