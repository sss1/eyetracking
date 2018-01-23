import numpy as np
from scipy.stats import multivariate_normal
from math import log

# sigma2 - spherical emission variance (i.e., E[||X - E[X]||_2^2])
def get_trackit_MLE(eye_track, target, distractors, sigma2 = 100 ** 2):

  X = eye_track.swapaxes(0, 1)
  mu = np.concatenate((target[None, :, :], distractors)).swapaxes(1, 2)
  return get_MLE(X, mu, sigma2 = sigma2)

def get_MLE(X, mu, sigma2 = 100 ** 2): 

  # For now, just hardcode reasonable guesses of model parameters
  n_states = mu.shape[0]
  trans_prob = (1.0/600.0)/(n_states - 1.0) # Probability of transitioning between any pair of states # was originally 0.0001
  # This guess corresponds to an average of 1 switches per 600 frames, or roughly once every 10 seconds
  pi = np.ones(n_states) / n_states # Uniform starting probabilities
  Pi = (1 - n_states*trans_prob) * np.identity(n_states) + trans_prob * np.ones((n_states,n_states))
  return __viterbi(X, mu, pi, Pi, sigma2 = sigma2)

# In the following,
#   N denotes the length (in frames) of the trial
#   K denotes the number of objects (number of distractors + 1 target)
#
# X is an N x 2 sequence of (x,y) eye-tracking pairs
# mu is a K x N x 2 matrix (x,y) object positions for each object
# pi is a K-vector of initial probabilities of each state
# Pi is a K x K matrix of transition probabilities between each pair of states
def __viterbi(X, mu, pi, Pi, sigma2 = 100 ** 2):

  N = X.shape[0]
  K = mu.shape[0]
  T = np.zeros((N, K)) # For each state at each timepoint, the maximum likelihood of any path to that state
  S = np.zeros((N - 1, K), dtype = np.int) # For each state, the most likely previous state

  # print 'N: ' + str(N) + ' K: ' + str(K) + ' T.shape: ' + str(T.shape) + ' S.shape: ' + str(S.shape)

  # For each state at each point in time, compute the maximum likelihood (over
  # paths) of ending up at that state
  # Note that, whenever X[n,:] is [nan, nan] (i.e., eye-tracking data is missing),
  # this sets T[n, k] = -inf and S[n - 1, :] = [-1,...,-1]
  for k in range(K): # First state likelhoods are based on starting probabilities
    T[0, k] = log(pi[k]) + log_emission_prob(X[0, :], mu[k, 0, :], sigma2 = sigma2)
  for n in range(1, N): # time step
    for k in range(K): # current state
      max_likelihood = float("-inf")
      max_idx = -1
      for j in range(K): # previous state
        # if previous sample is NaN and this one is valid (not NaN), then use initial probabilities pi
        if np.isnan(X[n - 1, 0]) and not np.isnan(X[n, 0]):
          assert(T[n - 1, j] == float("-inf"))
          next_likelihood = pi[k] + log_emission_prob(X[n, :], mu[k, n, :], sigma2 = sigma2)
        else:
          next_likelihood = T[n - 1, j] + log(Pi[j, k]) + log_emission_prob(X[n, :], mu[k, n, :], sigma2 = sigma2)
        if next_likelihood > max_likelihood:
          max_likelihood = next_likelihood
          max_idx = j
      T[n, k] = max_likelihood
      S[n - 1, k] = max_idx

  # Having computed all the most likely paths to each final states, extract the
  # most likely sequence of states
  MLE = np.zeros((N,), dtype = np.int) # Final most likely sequence of states
  for n in reversed(range(N)):
    if np.isnan(X[n, 0]):
      MLE[n] = -1
    elif n == N - 1 or np.isnan(X[n + 1, 0]): # last valid sample before trial end or an interval of NaNs
      MLE[n] = np.argmax(T[n, :])
    else:
      MLE[n] = S[n, MLE[n + 1]]
    if np.isnan(X[n, 0]):
      assert MLE[n] == -1
    else:
      assert MLE[n] > -1
  return MLE

def log_emission_prob(X, mu, sigma2 = 100 ** 2):
  # Add singleton dimension using None because log_multivariate_normal_density is written for
  # multiple samples, but we only need it for 1
  return multivariate_normal.logpdf(X, mean = mu, cov = sigma2)
