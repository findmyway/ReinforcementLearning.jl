# Tutorial
You would like to test existing reinforcement learning methods on your
environment or try your method on existing environments? Extending this package
is a piece of cake. Please consider registering the binding to your own
environment as a new package (see e.g.
[ReinforcementLearningEnvironmentAtari](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentAtari.jl)) and
open a [pull
request](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pulls)
for any other extension.

## Write your own learner

For a new learner you need to implement the functions
```
update!(learner, buffer)                          # returns nothing
selectaction(learner, policy, state)              # returns an action
defaultbuffer(learner, environment, preprocessor) # returns a buffer
```

Let's assume you want to implement plain, simple Q-learning (you don't need to
do this; it is already implemented. Your file `qlearning.jl` could contain
```julia
import ReinforcementLearning: update!, selectaction, defaultbuffer, Buffer

struct MyQLearning
    Q::Array{Float64, 2} # number of actions x number of states
    alpha::Float64       # learning rate
end

function update!(learner::MyQLearning, buffer)
    s = buffer.states[1]
    snext = buffer.states[2]
    r = buffer.rewards[1]
    a = buffer.actions[1]
    Q = learner.Q
    Q[a, s] += learner.alpha * (r + maximum(Q[:, snext]) - Q[a, s])
end

function selectaction(learner::MyQLearning, policy, state)
    selectaction(policy, learner.Q[:, state])
end

function defaultbuffer(learner::MyQLearning, environment, preprocessor)
    state, done = getstate(environment)
    processedstate = preprocessstate(preprocessor, state)
    Buffer(statetype = typeof(processedstate), capacity = 2)
end
```
The function `defaultbuffer` gets called during the construction of an
`RLSetup`. It returns a buffer that is filled with states, actions and rewards
during interaction with the environment. Currently there are three types of
Buffers implemented
```julia
import ReinforcementLearning: Buffer, EpisodeBuffer, ArrayStateBuffer
?Buffer
```

## [Bind your own environment](@id api_environments)
For new environments you need to implement the functions
```
interact!(action, environment)          # returns state, reward done
getstate(environment)                   # returns state, done
reset!(environment)                     # returns state
```

Optionally you may also implement the function
```
plotenv(environment, state, action, reward, done)
```

Please have a look at the
[cartpole](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentClassicControl.jl/blob/master/src/cartpole.jl)
for an example.

## Preprocessors
```
preprocessstate(preprocessor, state)    # returns the preprocessed state
```
Optional:
```
preprocess(preprocessor, reward, state, done) # returns a preprocessed (state, reward done) tuple.
```

## Policies
```
selectaction(policy, values)            # returns an action
getprob(policy, state)   # Returns a normalized (1-norm) vector with non-negative entries.
```

## Callbacks
```
callback!(callback, rlsetup, state, action, reward, done) # returns nothing
```

## Stopping Criteria
```
isbreak!(stoppingcriterion, state, action, reward, done) # returns true of false
```
