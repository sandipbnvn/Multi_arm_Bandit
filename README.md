# Multi Arm Bandit
Implementation of multi arm bandit in a 10 arm test bed

## Single Bandit problem
The bandit problem should have  10  arms. For each arm, the  q∗(a)  value should be sampled from a Gaussian distribution with  0  mean and unit variance . The rewards from arm  a  should be sampled from a Gaussian distribution with mean  q(a)  and unit variance.

## Testbed
Generate 2000 different bandit problems. Use the sample average method with incremental implementation for learning the action values. Plot the learning curves for 1000 timesteps with the following algorithms:

gready
ϵ-greedy
ϵ-0.1
ϵ-0.01
There are two learning curves for each algorithm:

average reward vs time
% optimal actions vs time
You have to reproduce the following two curves:

Average rewards vs # of steps
% optimal action vs # of steps

Hint: the averaging at each time step is done over all the 2000 bandit problems. For example, to plot a single point  (t,r(t))  in the "average reward plot", find the reward sampled from each of the 2000 bandits at time  t  and compute their average. This value will be  r(t) .

2. Run the UCB algorithm on the ten-armed testbed. Plot the learning curve for UCB and compare it with  ϵ -greedy for  ϵ=0.1 . Reproduce the following learning curves averaged over 2000 bandits for 1000 timesteps
