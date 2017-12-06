import numpy as np
# from hmmlearn import hmm
from load_data import load_full_subject_data
# import warnings
from viterbi import viterbi
import util

def getMLE(eye_track, target, distractors):
  # root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
  # # root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
  # subject_type = "adult_pilot/"
  # TI_data_dir = "TrackItOutput/AllSame/"
  # ET_data_dir = "EyeTracker/AllSame/"
  # TI_fname = "AnnaSame.csv"
  # ET_fname = "AnnaSame_9_13_2016_13_25.csv"
  # TI_file_path = root + subject_type + TI_data_dir + TI_fname
  # ET_file_path = root + subject_type + ET_data_dir + ET_fname
  # 
  # # print 'Track-It file: ' + TI_file_path
  # 
  # # track_it_xy_list and eye_track_xy_list are each 3D lists of shape (trial, axis (xy), time)
  # track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  #   = load_full_subject_data(TI_file_path,
  #                             ET_file_path,
  #                             filter_threshold = 1)
  # 
  # 
  # # Just do first trial, for now
  # trial_idx = 2
  # X = np.swapaxes(np.array(eye_track_xy_list[trial_idx]), 0, 1)
  # # Since the eyetracking data is higher frequency than the TrackIt data, but the TrackIt data is
  # # piecewise linear, we interpolate the latter so that both have the same number of timepoints
  # eyetrack_len = X.shape[0]
  # print('eyetrack_len: ' + str(eyetrack_len))

  # distractors_xy_list[trial_idx] = distractors_xy_list[trial_idx][0:num_distractors]
  # mu = np.zeros((num_distractors + 1, eyetrack_len, 2))
  # mu[0, :, 0] = util.interpolate_to_length(np.array(track_it_xy_list[trial_idx][0]), eyetrack_len)
  # mu[0, :, 1] = util.interpolate_to_length(np.array(track_it_xy_list[trial_idx][1]), eyetrack_len)
  # for k in range(num_distractors):
  #   mu[k + 1, :, 0] = util.interpolate_to_length(np.array(distractors_xy_list[trial_idx][k][0]), eyetrack_len)
  #   mu[k + 1, :, 1] = util.interpolate_to_length(np.array(distractors_xy_list[trial_idx][k][1]), eyetrack_len)

  # N_short = 700
 
  # X = X[0:N_short, :] # TODO: REMOVE THIS; JUST USED TO AVOID NANS FOR DEBUGGING!
  # print mu.shape
  # mu = mu[:, 0:N_short, :]
 
  X = eye_track.swapaxes(0, 1)
  mu = np.concatenate((target[None, :, :], distractors)).swapaxes(1, 2)
 
  # For now, just hardcode model parameters
  trans_prob = 0.0001 # Probability of transitioning between any pair of states
  n_states = mu.shape[0]
  pi = np.ones(n_states) / n_states # Uniform starting probabilities
  Pi = (1 - n_states*trans_prob) * np.identity(n_states) + trans_prob * np.ones((n_states,n_states))
  
  MLE = viterbi(X, mu, pi, Pi)
  return MLE, X, mu

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
