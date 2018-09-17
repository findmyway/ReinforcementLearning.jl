"""
    mutable struct MonteCarlo <: AbstractReinforcementLearner
        ns::Int64 = 10
        na::Int64 = 4
        γ::Float64 = .9
        initvalue = 0.
        Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
        Q::Array{Float64, 2} = zeros(na, ns) + initvalue


Estimate Q values by averaging over returns.
"""
@with_kw struct MonteCarlo
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    initvalue = 0.
    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
    Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
end
function defaultbuffer(learner::MonteCarlo, env, preprocessor)
    EpisodeTurnBuffer{typeof(getstate(env).observation), typeof(actionspace(env)), Float64, Bool}()
end

export MonteCarlo

function update!(learner::MonteCarlo, buffer)
    t_end = buffer[end]
    if learner.Q[t_end.action, t_end.state] == Inf64
        learner.Q[t_end.action, t_end.state] = 0.
    end
    if buffer.isdone[end]
        G = 0.
        for t in length(buffer):-1:1
            turn = buffer[t]
            G = learner.γ * G + turn.reward
            n = learner.Nsa[turn.action, turn.state] += 1
            learner.Q[turn.action, turn.state] *= (1 - 1/n)
            learner.Q[turn.action, turn.state] += 1/n * G
        end
    end
end
