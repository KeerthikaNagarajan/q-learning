# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.


## PROBLEM STATEMENT
Design and implement a Python program that employs the Q-Learning algorithm to determine the optimal policy for a given Reinforcement Learning (RL) environment. Additionally, compare the state values obtained using Q-Learning with those obtained using the Monte Carlo method for the same RL environment.
## Q LEARNING ALGORITHM
Step 1:
Initialize Q-table and hyperparameters.

Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

Step 3:
After training, derive the optimal policy from the Q-table.

Step 4:
Implement the Monte Carlo method to estimate state values.

Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.
## Q LEARNING FUNCTION
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
### Optimal State Value Functions:
<img width="671" alt="image" src="https://github.com/KeerthikaNagarajan/q-learning/assets/93427089/08c43562-8e91-4f35-9ed2-1f556b5f1b53">


### Optimal Action Value Functions:

<img width="656" alt="image" src="https://github.com/KeerthikaNagarajan/q-learning/assets/93427089/02f9fd14-f0c3-4113-be56-40d8abcf49a0">

### State value functions of Monte Carlo method:
<img width="868" alt="image" src="https://github.com/KeerthikaNagarajan/q-learning/assets/93427089/7e02fbe5-b376-4bae-8d5c-f843baa373b6">

### State value functions of Qlearning method:

<img width="866" alt="image" src="https://github.com/KeerthikaNagarajan/q-learning/assets/93427089/e03ebf7c-cae4-47b7-899b-81c61e59c85e">

## RESULT:

Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
