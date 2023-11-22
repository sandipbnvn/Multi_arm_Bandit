'''
Run the UCB algorithm on the ten-armed testbed. Plot the learning curve for UCB and compare it with  ϵ -greedy for  ϵ=0.1 . 
Reproduce the following learning curves averaged over 2000 bandits for 1000 timesteps:
'''

# Constants
num_bandits = 10
num_problems = 2000
num_steps = 1000
epsilon = 0.1
c = 2  # Exploration parameter for UCB

# Generate bandit problems
np.random.seed(42)
true_action_values = np.random.normal(loc=0, scale=1, size=(num_problems, num_bandits))

# UCB algorithm
def ucb_algorithm(Q_values, action_counts, step):
    ucb_values = Q_values + c * np.sqrt(np.log(step + 1) / (action_counts + 1e-6))
    return np.argmax(ucb_values)

# Epsilon-greedy algorithm
def epsilon_greedy(Q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q_values))
    else:
        return np.argmax(Q_values)

# Function to run bandit experiments for average reward
def run_bandit_experiment(ucb=False):
    average_rewards = np.zeros(num_steps)

    for problem in range(num_problems):
        Q_values = np.zeros(num_bandits)
        action_counts = np.zeros(num_bandits)

        for step in range(num_steps):
            if ucb:
                action = ucb_algorithm(Q_values, action_counts, step)
            else:
                action = epsilon_greedy(Q_values, epsilon)

            reward = np.random.normal(loc=true_action_values[problem, action], scale=1)
            action_counts[action] += 1
            Q_values[action] += (reward - Q_values[action]) / action_counts[action]

            average_rewards[step] += reward

    average_rewards /= num_problems
    return average_rewards

# Run experiments for UCB and epsilon-greedy (epsilon=0.1) for average reward
ucb_rewards = run_bandit_experiment(ucb=True)
epsilon_greedy_rewards = run_bandit_experiment()

# Plotting average rewards for both algorithms
plt.figure(figsize=(12, 6))
plt.plot(ucb_rewards, label='UCB c=2', color='b')
plt.plot(epsilon_greedy_rewards, label='$\epsilon$-Greedy (Epsilon=0.1)', color='grey')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Average Reward Comparison: UCB vs Epsilon-Greedy')
plt.show()
