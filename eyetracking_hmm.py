import numpy as np
from hmmlearn import hmm
from load_data import load_full_subject_data
import warnings

root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/"
subject_type = "adult_pilot/"
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "AnnaSame.csv"
ET_fname = "AnnaSame_9_13_2016_13_25.csv"
TI_file_path = root + subject_type + TI_data_dir + TI_fname
ET_file_path = root + subject_type + ET_data_dir + ET_fname

print 'Track-It file: ' + TI_file_path

track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

n_states = len(distractors_xy_list[0]) + 1 # number of distractors + 1

print n_states

trans_prob = 0.05 # Probability of transitioning between any pair of states

# MODEL COPIED FROM EXAMPLE
model = hmm.GaussianHMM(n_components = n_states, covariance_type = "full")
model.startprob_ = np.ones(n_states) / n_states # Uniform starting probabilities
model.transmat_ = (1 - n_states*trans_prob) * np.identity(n_states) + trans_prob * np.ones((n_states,n_states))
model.means_ = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0]])
model.covars_ = np.tile(np.identity(2), (5, 1, 1))
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=DeprecationWarning)
  X, Z = model.sample(10000) # We will want to replace this with model.fit
print X
print Z
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=DeprecationWarning)
  print model.fit(X)
