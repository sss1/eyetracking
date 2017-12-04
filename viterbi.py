import numpy as np
from scipy.stats import multivariate_normal
from stats import log_multivariate_normal_density
from math import log

sigma2 = 30 ** 2 # spherical emission variance (i.e., E[||X - E[X]||_2^2])

# In the following,
#   N denotes the length (in frames) of the trial
#   K denotes the number of objects (number of distractors + 1 target)
#
# X is an N x 2 sequence of (x,y) eye-tracking pairs
# mu is a K x N x 2 matrix (x,y) object positions for each object
# pi is a K-vector of initial probabilities of each state
# Pi is a K x K matrix of transition probabilities between each pair of states
def viterbi(X, mu, pi, Pi):

  N = X.shape(0)
  K = mu.shape(0)
  T = np.zeros((N, K)) # For each state at each timepoint, the maximum likelihood of any path to that state
  S = np.zeros((N - 1, K), dtype = np.int) # For each state, the most likely previous state

  # For each state at each point in time, compute the maximum likelihood (over
  # paths) of ending up at that state
  for k in range(K): # First state likelhoods are based on starting probabilities
    T[0, k] = log(pi[k]) + emission_prob(X[0, :], mu[0, k])
  for n in range(1, N): # time step
    for k in range(K): # current state
      max_likelihood = float("-inf")
      max_idx = -1
      for j in range(K): # previous state
        next_likelihood = T[n - 1, j] + log(Pi[j, k]) + emission_prob(X[n, :], mu[n, k])
        if next_likelihood > max_likelihood:
          max_likelihood = next_likelihood
          max_idx = j
      T[n, k] = max_likelihood
      S[n - 1, k] = max_idx

  # Having computed all the most likely paths to each final states, extract the
  # most likely sequence of states
  MLE = np.zeros((N - 1, K), dtype = np.int) # Final most likely sequence of states
  MLE[N] = argmax(T[N, :])
  for n in reversed(range(N - 1)):
    MLE[n] = S[n, X[n + 1]]

def emission_prob(X, mu):
  return log_multivariate_normal_density(X, mu, sigma2 * np.identify(2), covariance_type = 'diag')
