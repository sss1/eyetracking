from load_data import load_full_subject_data
from eyetracking_hmm import get_trackit_MLE
import matplotlib.pyplot as plt
from util import preprocess_all, jagged_to_numpy
import data_paths as dp
import timeit
import time
import numpy as np

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

# For each experiment type, plot histogram of missing data proportion over trials
# plt.figure(0)
# plt.subplot(3, 2, 1)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_adult_0dis for trial_data in subject_data[0]])
# plt.title('Adult, 0 distractors')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.subplot(3, 2, 3)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_adult_same for trial_data in subject_data[0]])
# plt.title('Adult, All Same')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.subplot(3, 2, 5)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_adult_diff for trial_data in subject_data[0]])
# plt.title('Adult, All Different')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.subplot(3, 2, 2)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_child_0dis for trial_data in subject_data[0]])
# plt.title('Child, 0 distractors')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.subplot(3, 2, 4)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_child_same for trial_data in subject_data[0]])
# plt.title('Child, All Same')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.subplot(3, 2, 6)
# plt.hist([np.isnan(trial_data[0,:]).mean() for subject_data in data_child_diff for trial_data in subject_data[0]])
# plt.title('Child, All Different')
# plt.xlim(0, 1)
# plt.ylim(0, 50)
# 
# plt.gcf().suptitle('Distribution of Proportion of Missing Data per Trial', fontsize = "x-large")
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

# For a range of thresholds between 0 and 1, plot histogram of trials per subject with >= threshold proportion of valid data
# plt.figure(1)
# data = data_child_diff
# condition_name = 'All Different'
# num_plots = 10
# x_max = max([len(subject_data[0]) for subject_data in data]) + 1
# for threshold_idx in range(num_plots):
#   threshold = threshold_idx / float(num_plots)
#   print threshold
#   plt.subplot(num_plots, 1, threshold_idx + 1)
#   plt.hist([sum([np.mean(np.isnan(trial_data[0,:])) > threshold for trial_data in subject_data[0]]) for subject_data in data], bins = [x/2.0 for x in range(2*x_max)])
#   plt.xlim(0, x_max)
#   plt.ylim(0, len(data_child_0dis))
#   plt.ylabel(str(threshold))
# plt.gcf().text(0.04, 0.5, 'Threshold', va = 'center', rotation = 'vertical')
# plt.gcf().suptitle('Distribution of Trials per Subject\nwith Missing Data Proportion at most \"Thresholding\"\n(' + condition_name + ' Condition)', fontsize = "x-large")
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

# TODO: Once we've added an inattentive state, run MLEs for 0dis condition too
# Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
MLEs_adult_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_same]
# print np.array([trial_data[0] for subject_data in MLEs_adult_same for trial_data in subject_data])
# print np.array([(trial_data[0] == 0) for subject_data in MLEs_adult_same for trial_data in subject_data]).mean()

MLEs_adult_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_diff]
# print np.array([trial_data[0] for subject_data in MLEs_adult_diff for trial_data in subject_data])
# print np.array([(trial_data[0] == 0) for subject_data in MLEs_adult_diff for trial_data in subject_data]).mean()
# 
# MLEs_child_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_same]
# print np.array([trial_data[0] for subject_data in MLEs_child_same for trial_data in subject_data])
# print np.array([(trial_data[0] == 0) for subject_data in MLEs_child_same for trial_data in subject_data]).mean()
# 
# MLEs_child_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_diff]
# print np.array([trial_data[0] for subject_data in MLEs_child_diff for trial_data in subject_data])
# print np.array([(trial_data[0] == 0) for subject_data in MLEs_child_diff for trial_data in subject_data]).mean()

# TODO: Compare same and different conditions; is state == 0 much more often for same than for different?
# No need to continue distinguishing trials from different subjects; replace state with 0 if on target, 1 else
aligned = jagged_to_numpy([[float(x != 0) for x in trial_data] for subject_data in MLEs_adult_same for trial_data in subject_data])

print 'aligned.shape: ' + str(aligned.shape)
print 'np.nanmean(aligned, axis = 0).shape: ' + str(np.nanmean(aligned, axis = 0).shape)
print 'np.nanmean(aligned, axis = 0): ' + str(np.nanmean(aligned, axis = 0))

















