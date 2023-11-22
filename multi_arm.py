import numpy as np
import matplotlib.pyplot as plt

# Constants
num_bandits = 10
num_problems = 2000
num_steps = 1000

# Generate bandit problems
np.random.seed(42)
true_action_values = np.random.normal(loc=0, scale=1, size=(num_problems, num_bandits))

# Helper function for epsilon-greedy action selection
def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))
    else:
        return np.argmax(q_values)

# Function to run bandit experiments
def run_bandit_experiment(epsilon):
    average_rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps)

    for problem in range(num_problems):
        q_values = np.zeros(num_bandits)
        action_counts = np.zeros(num_bandits)

        for step in range(num_steps):
            action = epsilon_greedy(q_values, epsilon)
            reward = np.random.normal(loc=true_action_values[problem, action], scale=1)
            action_counts[action] += 1
            q_values[action] += (reward - q_values[action]) / action_counts[action] #updating q values in incremental fashion

            average_rewards[step] += reward
            optimal_actions[step] += 1 if action == np.argmax(true_action_values[problem]) else 0

    average_rewards /= num_problems
    optimal_actions = (optimal_actions / num_problems) * 100

    return average_rewards, optimal_actions

# Run experiments for different algorithms
epsilons = [0, 0.01, 0.1]
algorithm_names = ['Greedy', '$\epsilon$-greedy=0.01', '$\epsilon$-greedy=0.1']
colors = ['g', 'r', 'b']

plt.figure(figsize=(12, 10))

for epsilon, name, color in zip(epsilons, algorithm_names, colors):
    avg_rewards, optimal_actions = run_bandit_experiment(epsilon)
    plt.subplot(2, 1, 1)
    plt.title("Average Reward over Time")
    plt.plot(avg_rewards, label=name, color=color)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("% of Optimal Action over Time")
    plt.plot(optimal_actions, label=name, color=color)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()

plt.tight_layout()
plt.show()
