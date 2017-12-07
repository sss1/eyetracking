import math
import numpy as np

# Euclidean distance error
def error_fcn(et_x, et_y, trackit_x, trackit_y):
    return math.sqrt((et_x - trackit_x)**2 + (et_y - trackit_y)**2)

# Data preprocessing steps: impute missing eye-tracking data and synchronize by interpolating TrackIt data
def preprocess_all(eyetrack, target, distractors):
  for trial_idx in range(len(eyetrack)):
    N = len(eyetrack[trial_idx][0]) # Number of eye-tracking frames in trial
    eyetrack[trial_idx] = impute_missing_data_D(eyetrack[trial_idx])
    target[trial_idx] = interpolate_to_length_D(target[trial_idx], N)
    distractors[trial_idx] = interpolate_to_length_distractors(distractors[trial_idx], N)
  return eyetrack, target, distractors

# Given a K x D x N array of numbers, encoding the positions of each of K D-dimensional objects over N time points,
# performs interpolate_to_length_D (independently) on each object in X
def interpolate_to_length_distractors(X, new_len):
  K = X.shape[0]
  D = X.shape[1]
  X_new = np.zeros((K, D, new_len))
  for k in range(K):
    X_new[k,:,:] = interpolate_to_length_D(X[k,:,:], new_len)
  return X_new

# Given a D-dimensional sequence X of numbers, performs interpolate_to_length (independently) on each dimension of X
# X is D x N, where D is the dimensionality and N is the sample length
def interpolate_to_length_D(X, new_len):
  D = X.shape[0]
  X_new = np.zeros((D, new_len))
  for d in range(D):
    X_new[d, :] = __interpolate_to_length(X[d, :], new_len)
  return X_new

# Given a sequence X of numbers, returns the length-new_len linear interpolant of X
def __interpolate_to_length(X, new_len):
  old_len = X.shape[0]
  return np.interp([(float(n)*old_len)/new_len for n in range(new_len)], range(old_len), X)

# Given a D-dimensional sequence X of numbers, performs impute_missing_data_D (independently) on each dimension of X
# X is D x N, where D is the dimensionality and N is the sample length
def impute_missing_data_D(X, max_len = 10):
  D = X.shape[0]
  for d in range(D):
    X[d, :] = __impute_missing_data(X[d, :], max_len)
  return X

# Given a sequence X of floats, replaces short streches (up to length max_len) of NaNs with linear interpolation
#
# For example, if
#
# X = np.array([1, NaN, NaN,  4, NaN,  6])
#
# then
#
# impute_missing_data(X, max_len = 1) == np.array([1, NaN, NaN, 5, 6])
#
# and
#
# impute_missing_data(X, max_len = 2) == np.array([1, 2, 3, 4, 5, 6])
def __impute_missing_data(X, max_len):
  last_valid_idx = -1
  last_valid_val = float('nan')
  for n in range(len(X)):
    if not math.isnan(X[n]):
      if last_valid_idx < n - 1: # there is missing data
        if n - (max_len + 1) <= last_valid_idx: # amount of missing data is at most than max_len
          first_last = np.array([X[last_valid_idx], X[n]]) # initial and final values from which to linearly interpolate
          new_len = n - last_valid_idx + 1
          X[last_valid_idx:(n + 1)] = np.interp([float(x)/(new_len - 1) for x in range(new_len)], [0, 1], first_last)
          # print('Successfully interpolated a block of ' + str(new_len - 2) + ' NaN(s).')
        else:
          new_len = n - last_valid_idx + 1
          print('Failed to interpolate a block of ' + str(new_len - 2) + ' NaN(s).')
      last_valid_idx = n
      last_valid_val = X[n]
  return X

# X = np.array([1, 2, 3])
# print __impute_missing_data(X, 10)
# X = np.array([0, 1])
# print __interpolate_fixed_length(X, 5)
# X = np.array([1, float('nan'), float('nan'), float('nan'), 5, float('nan'), float('nan'), 8])
# print X
# print __impute_missing_data(X, 3)
