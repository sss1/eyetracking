from load_data import load_full_subject_data
from eyetracking_hmm import get_trackit_MLE
import matplotlib.pyplot as plt
from util import preprocess_all, jagged_to_numpy
import data_paths as dp
import timeit
import time
import numpy as np

# load datasets
# data is (subject X (eyetrack/target/distractors) X trial)
data_adult_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_0dis, dp.eyetrack_fnames_adult_0dis)]
data_adult_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_same, dp.eyetrack_fnames_adult_same)]
data_adult_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_diff, dp.eyetrack_fnames_adult_diff)]
data_child_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_0dis, dp.eyetrack_fnames_child_0dis)]
data_child_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_same, dp.eyetrack_fnames_child_same)]
data_child_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_diff, dp.eyetrack_fnames_child_diff)]

print '\nMissing data before interpolation:'
print 'Adult Same: ' + str(np.mean(np.isnan([x for subject_data in data_adult_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult Diff: ' + str(np.mean(np.isnan([x for subject_data in data_adult_diff for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Same: ' + str(np.mean(np.isnan([x for subject_data in data_child_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Diff: ' + str(np.mean(np.isnan([x for subject_data in data_child_diff for trial_data in subject_data[0] for x in trial_data[0]])))

# Preprocess data (synchronize TrackIt with eyetracking, and interpolate some missing data
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

sigma2_child = 421.696503429 ** 2 # Value taken from supervised analysis

print '\nMissing data after interpolation:'
print 'Adult Same: ' + str(np.mean(np.isnan([x for subject_data in data_adult_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult Diff: ' + str(np.mean(np.isnan([x for subject_data in data_adult_diff for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Same: ' + str(np.mean(np.isnan([x for subject_data in data_child_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Diff: ' + str(np.mean(np.isnan([x for subject_data in data_child_diff for trial_data in subject_data[0] for x in trial_data[0]])))

# # Apply HMM analysis to each dataset
# # Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
# # TODO: Once we've added an inattentive state, run MLEs for 0dis condition too
# MLEs_adult_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_same]
# MLEs_adult_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_adult_diff]
# MLEs_child_same = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_same]
# MLEs_child_diff = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_diff]
# # Since HMM analysis is slow, cache the results
# np.savez(dp.root + 'tmp/' + 'MLEs_all.npz', \
#          MLEs_adult_same = MLEs_adult_same, \
#          MLEs_adult_diff = MLEs_adult_diff, \
#          MLEs_child_same = MLEs_child_same, \
#          MLEs_child_diff = MLEs_child_diff)

MLEs_cache_file = np.load(dp.root + 'cache/' + 'MLEs_all.npz')

MLEs_adult_same = MLEs_cache_file['MLEs_adult_same']
MLEs_adult_diff = MLEs_cache_file['MLEs_adult_diff']
MLEs_child_same = MLEs_cache_file['MLEs_child_same']
MLEs_child_diff = MLEs_cache_file['MLEs_child_diff']

print '\nProportion of time (total frames) on target:'
print 'Adult Same: ' + str(np.mean([x == 0 for subject_data in MLEs_adult_same for trial_data in subject_data for x in trial_data]))
print 'Adult Diff: ' + str(np.mean([x == 0 for subject_data in MLEs_adult_diff for trial_data in subject_data for x in trial_data]))
print 'Child Same: ' + str(np.mean([x == 0 for subject_data in MLEs_child_same for trial_data in subject_data for x in trial_data]))
print 'Child Diff: ' + str(np.mean([x == 0 for subject_data in MLEs_child_diff for trial_data in subject_data for x in trial_data]))

print '\nProportion of time (non-missing or interpolated frames) on target:'
print 'Adult Same: ' + str(np.mean([x == 0 for subject_data in MLEs_adult_same for trial_data in subject_data for x in trial_data[trial_data > -1]]))
print 'Adult Diff: ' + str(np.mean([x == 0 for subject_data in MLEs_adult_diff for trial_data in subject_data for x in trial_data[trial_data > -1]]))
print 'Child Same: ' + str(np.mean([x == 0 for subject_data in MLEs_child_same for trial_data in subject_data for x in trial_data[trial_data > -1]]))
print 'Child Diff: ' + str(np.mean([x == 0 for subject_data in MLEs_child_diff for trial_data in subject_data for x in trial_data[trial_data > -1]]))


# No need to continue distinguishing trials from different subjects; replace state with 0 if on target, 1 else
# Plot proportion of trials on target over trial time
plt.figure(1)
aligned_adult_same = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_adult_same for trial_data in subject_data])
aligned_adult_diff = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_adult_diff for trial_data in subject_data])
aligned_child_same = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_child_same for trial_data in subject_data])
aligned_child_diff = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_child_diff for trial_data in subject_data])

plt.plot(np.nanmean(aligned_adult_same, axis = 0))
plt.plot(np.nanmean(aligned_adult_diff, axis = 0))
plt.plot(np.nanmean(aligned_child_same, axis = 0))
plt.plot(np.nanmean(aligned_child_diff, axis = 0))

plt.legend(['Adult Same', 'Adult Diff', 'Child Same', 'Child Diff'])


# For each dataset, plot a vertical line indicating the length of the shortest trial
min_trial_len_adult_same = min([len(trial_data) for subject_data in MLEs_adult_same for trial_data in subject_data])
min_trial_len_adult_diff = min([len(trial_data) for subject_data in MLEs_adult_diff for trial_data in subject_data])
min_trial_len_child_same = min([len(trial_data) for subject_data in MLEs_child_same for trial_data in subject_data])
min_trial_len_child_diff = min([len(trial_data) for subject_data in MLEs_child_diff for trial_data in subject_data])

plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_adult_same, min_trial_len_adult_same], [0, 1])
plt.plot([min_trial_len_adult_diff, min_trial_len_adult_diff], [0, 1])
plt.plot([min_trial_len_child_same, min_trial_len_child_same], [0, 1])
plt.plot([min_trial_len_child_diff, min_trial_len_child_diff], [0, 1])

plt.xlabel('Time (frames, at 60Hz)')
plt.ylabel('Fraction of all trials on target')


# Plot proportion of (non-missing or interpolated) trials on target over trial time
plt.figure(2)

def on_target_or_nan(x):
  if x == -1:
    return float("nan")
  return x == 0

aligned_adult_same = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_adult_same for trial_data in subject_data])
aligned_adult_diff = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_adult_diff for trial_data in subject_data])
aligned_child_same = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_child_same for trial_data in subject_data])
aligned_child_diff = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_child_diff for trial_data in subject_data])

plt.plot(np.nanmean(aligned_adult_same, axis = 0))
plt.plot(np.nanmean(aligned_adult_diff, axis = 0))
plt.plot(np.nanmean(aligned_child_same, axis = 0))
plt.plot(np.nanmean(aligned_child_diff, axis = 0))

plt.legend(['Adult Same', 'Adult Diff', 'Child Same', 'Child Diff'])

plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_adult_same, min_trial_len_adult_same], [0, 1])
plt.plot([min_trial_len_adult_diff, min_trial_len_adult_diff], [0, 1])
plt.plot([min_trial_len_child_same, min_trial_len_child_same], [0, 1])
plt.plot([min_trial_len_child_diff, min_trial_len_child_diff], [0, 1])

plt.xlabel('Time (frames, at 60Hz)')
plt.ylabel('Fraction of non-missing or interpolated trials on target')


# Plot proportion of missing data over trial time
plt.figure(3)
aligned_adult_same = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_adult_same for trial_data in subject_data])
aligned_adult_diff = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_adult_diff for trial_data in subject_data])
aligned_child_same = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_child_same for trial_data in subject_data])
aligned_child_diff = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_child_diff for trial_data in subject_data])

plt.plot(np.nanmean(aligned_adult_same, axis = 0))
plt.plot(np.nanmean(aligned_adult_diff, axis = 0))
plt.plot(np.nanmean(aligned_child_same, axis = 0))
plt.plot(np.nanmean(aligned_child_diff, axis = 0))

plt.legend(['Adult Same', 'Adult Diff', 'Child Same', 'Child Diff'])

plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_adult_same, min_trial_len_adult_same], [0, 1])
plt.plot([min_trial_len_adult_diff, min_trial_len_adult_diff], [0, 1])
plt.plot([min_trial_len_child_same, min_trial_len_child_same], [0, 1])
plt.plot([min_trial_len_child_diff, min_trial_len_child_diff], [0, 1])

plt.xlabel('Trial Time (frames, at 60Hz)')
plt.ylabel('Fraction of trials missing data')

plt.show()










