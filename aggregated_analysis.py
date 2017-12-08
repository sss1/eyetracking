from load_data import load_full_subject_data
from eyetracking_hmm import get_trackit_MLE
import matplotlib.pyplot as plt
from util import preprocess_all
import data_paths as dp
import timeit
import time
import numpy as np

# # CODE FOR LOADING DATA
# # root = "/home/sss1/Desktop/projects/eyetracking/data/" # Office desktop
# root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
# # root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
# subject_type = "adult_pilot/" # "3yo/" 
# TI_data_dir = "TrackItOutput/AllSame/"
# ET_data_dir = "EyeTracker/AllSame/"
# TI_fname = "AnnaSame.csv" # "shashank.csv" # "A232Same.csv" 
# ET_fname = "AnnaSame_9_13_2016_13_25.csv" # "shashank1_12_5_2017_13_41.csv" # "A232Same_3_29_2016_10_26.csv"
# 
# TI_file_path = root + subject_type + TI_data_dir + TI_fname
# ET_file_path = root + subject_type + ET_data_dir + ET_fname
# print 'Track-It file: ' + TI_file_path

# eyetrack_all_trials, target_all_trials, distractors_all_trials \
#   = load_full_subject_data(TI_file_path,
#                             ET_file_path,
#                             filter_threshold = 1)

# data is (subject X (eyetrack/target/distractors) X trial)
data_adult_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_0dis, dp.eyetrack_fnames_adult_0dis)]
data_adult_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_same, dp.eyetrack_fnames_adult_same)]
data_adult_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_diff, dp.eyetrack_fnames_adult_diff)]
data_child_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_0dis, dp.eyetrack_fnames_child_0dis)]
data_child_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_same, dp.eyetrack_fnames_child_same)]
data_child_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_diff, dp.eyetrack_fnames_child_diff)]

data_adult_0dis = [preprocess_all(*subject_data) for subject_data in data_adult_0dis]
data_adult_same = [preprocess_all(*subject_data) for subject_data in data_adult_same]
data_adult_diff = [preprocess_all(*subject_data) for subject_data in data_adult_diff]
data_child_0dis = [preprocess_all(*subject_data) for subject_data in data_child_0dis]
data_child_same = [preprocess_all(*subject_data) for subject_data in data_child_same]
data_child_diff = [preprocess_all(*subject_data) for subject_data in data_child_diff]

plt.subplot(3, 2, 1)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_adult_0dis for trial_data in subject_data[0]])
plt.title('Adult, 0 distractors')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.subplot(3, 2, 3)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_adult_same for trial_data in subject_data[0]])
plt.title('Adult, All Same')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.subplot(3, 2, 5)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_adult_diff for trial_data in subject_data[0]])
plt.title('Adult, All Different')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.subplot(3, 2, 2)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_child_0dis for trial_data in subject_data[0]])
plt.title('Child, 0 distractors')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.subplot(3, 2, 4)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_child_same for trial_data in subject_data[0]])
plt.title('Child, All Same')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.subplot(3, 2, 6)
plt.hist([np.mean(np.isnan(trial_data[0,:])) for subject_data in data_child_diff for trial_data in subject_data[0]])
plt.title('Child, All Different')
plt.xlim(0, 1)
plt.ylim(0, 50)

plt.gcf().suptitle('Proportion of Missing Data per Trial', fontsize = "x-large")
plt.show()


for subject_data in data_child_0dis:
  trial_means = [np.mean(np.isnan(trial_data[0,:])) for trial_data in subject_data[0]]
  print np.mean(trial_means)

# # Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
# def time_test1():
#   MLEs_adult_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_same]
# def time_test2():
#   MLEs_adult_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_diff]
# def time_test3():
#   MLEs_child_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_same]
# def time_test4():
#   MLEs_child_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_diff]
# print timeit.timeit(time_test1, number = 1)
# print timeit.timeit(time_test2, number = 1)
# print timeit.timeit(time_test3, number = 1)
# print timeit.timeit(time_test4, number = 1)
