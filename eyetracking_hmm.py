import numpy as np
# from hmmlearn import hmm
from load_data import load_full_subject_data
# import warnings
from viterbi import viterbi
import util

root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
# root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
subject_type = "adult_pilot/"
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "AnnaSame.csv"
ET_fname = "AnnaSame_9_13_2016_13_25.csv"
TI_file_path = root + subject_type + TI_data_dir + TI_fname
ET_file_path = root + subject_type + ET_data_dir + ET_fname

# print 'Track-It file: ' + TI_file_path

# track_it_xy_list and eye_track_xy_list are each 3D lists of shape (trial, axis (xy), time)
track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

# Just do first trial, with only target, for now
X = np.swapaxes(np.array(eye_track_xy_list[0]), 0, 1)

# Since the eyetracking data is higher frequency than the TrackIt data, but the TrackIt data is
# piecewise linear, we interpolate the latter so that both have the same number of timepoints
eyetrack_len = X.shape[0]
mu = np.zeros((eyetrack_len, 2))
mu[:, 0] = util.interpolate_to_length(np.array(track_it_xy_list[0][0]), eyetrack_len)
mu[:, 1] = util.interpolate_to_length(np.array(track_it_xy_list[0][1]), eyetrack_len)
mu = mu[None, 0:X.shape[0], :]

# Hardcode model parameters
trans_prob = 0.05 # Probability of transitioning between any pair of states
n_states = 1
pi = np.ones(n_states) / n_states # Uniform starting probabilities
Pi = (1 - n_states*trans_prob) * np.identity(n_states) + trans_prob * np.ones((n_states,n_states))

MLE = viterbi(X, mu, pi, Pi)
















# # MODEL COPIED FROM EXAMPLE
# n_states = len(distractors_xy_list[0]) + 1 # number of distractors + 1
# model = hmm.GaussianHMM(n_components = n_states, covariance_type = "full")
# model.startprob_ = np.ones(n_states) / n_states # Uniform starting probabilities
# model.transmat_ = (1 - n_states*trans_prob) * np.identity(n_states) + trans_prob * np.ones((n_states,n_states))
# model.means_ = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]])
# model.covars_ = np.tile(np.identity(2), (5, 1, 1))
# with warnings.catch_warnings():
#   warnings.filterwarnings("ignore",category=DeprecationWarning)
#   X, Z = model.sample(30) # We will want to replace this with model.predict
# print X
# print Z
# with warnings.catch_warnings():
#   warnings.filterwarnings("ignore",category=DeprecationWarning)
#   print model.predict(X)
