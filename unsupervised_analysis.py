from load_data import load_full_subject_data
import eyetracking_hmm
import naive_eyetracking
import matplotlib.pyplot as plt
from util import preprocess_all, jagged_to_numpy
import data_paths_COGSCI18 as dp # Pilot and supervised dataset for CogSci18 paper
import timeit
import time
import numpy as np

# load datasets
# data is (subject X (eyetrack/target/distractors) X trial)
# data_adult_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_0dis, dp.eyetrack_fnames_adult_0dis)]
data_adult_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_same, dp.eyetrack_fnames_adult_same)]
data_adult_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_adult_diff, dp.eyetrack_fnames_adult_diff)]
# data_child_0dis = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_0dis, dp.eyetrack_fnames_child_0dis)]
data_child_same = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_same, dp.eyetrack_fnames_child_same)]
data_child_diff = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_diff, dp.eyetrack_fnames_child_diff)]

print '\nMissing data before interpolation:'
print 'Adult Same: ' + str(np.mean(np.isnan([x for subject_data in data_adult_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult Diff: ' + str(np.mean(np.isnan([x for subject_data in data_adult_diff for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Same: ' + str(np.mean(np.isnan([x for subject_data in data_child_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Diff: ' + str(np.mean(np.isnan([x for subject_data in data_child_diff for trial_data in subject_data[0] for x in trial_data[0]])))

# Preprocess data (synchronize TrackIt with eyetracking, and interpolate some missing data
# data_adult_0dis = [preprocess_all(*subject_data) for subject_data in data_adult_0dis]
data_adult_same = [preprocess_all(*subject_data) for subject_data in data_adult_same]
data_adult_same = [subject_data for subject_data in data_adult_same if subject_data[0] is not None]
data_adult_diff = [preprocess_all(*subject_data) for subject_data in data_adult_diff]
data_adult_diff = [subject_data for subject_data in data_adult_diff if subject_data[0] is not None]
# data_child_0dis = [preprocess_all(*subject_data) for subject_data in data_child_0dis]
data_child_same = [preprocess_all(*subject_data) for subject_data in data_child_same]
data_child_same = [subject_data for subject_data in data_child_same if subject_data[0] is not None]
data_child_diff = [preprocess_all(*subject_data) for subject_data in data_child_diff]
data_child_diff = [subject_data for subject_data in data_child_diff if subject_data[0] is not None]

# Values taken from supervised analysis
sigma2_adult = 490 ** 2
sigma2_child = 870 ** 2

print '\nMissing data after interpolation:'
print 'Adult Same: ' + str(np.mean(np.isnan([x for subject_data in data_adult_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult Diff: ' + str(np.mean(np.isnan([x for subject_data in data_adult_diff for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Same: ' + str(np.mean(np.isnan([x for subject_data in data_child_same for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Child Diff: ' + str(np.mean(np.isnan([x for subject_data in data_child_diff for trial_data in subject_data[0] for x in trial_data[0]])))

# # Apply HMM analysis to each dataset
# # Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
# MLEs_adult_same = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_adult) for trial_data in zip(*subject_data)] for subject_data in data_adult_same]
# MLEs_adult_diff = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_adult) for trial_data in zip(*subject_data)] for subject_data in data_adult_diff]
# MLEs_child_same = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_child) for trial_data in zip(*subject_data)] for subject_data in data_child_same]
# MLEs_child_diff = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_child) for trial_data in zip(*subject_data)] for subject_data in data_child_diff]
# # Since HMM analysis is slow, cache the results
# np.savez(dp.root + 'cache/' + 'MLEs_all.npz', \
#          MLEs_adult_same = MLEs_adult_same, \
#          MLEs_adult_diff = MLEs_adult_diff, \
#          MLEs_child_same = MLEs_child_same, \
#          MLEs_child_diff = MLEs_child_diff)

MLEs_cache_file = np.load(dp.root + 'cache/' + 'MLEs_all.npz')

MLEs_adult_same = MLEs_cache_file['MLEs_adult_same']
MLEs_adult_diff = MLEs_cache_file['MLEs_adult_diff']
MLEs_child_same = MLEs_cache_file['MLEs_child_same']
MLEs_child_diff = MLEs_cache_file['MLEs_child_diff']

# Calculate TrackIt performance (proportion of frames on target) for each trial
trial_accuracies_adult_same = [np.mean(trial_data == 0) for subject_data in MLEs_adult_same for trial_data in subject_data]
trial_accuracies_adult_diff = [np.mean(trial_data == 0) for subject_data in MLEs_adult_diff for trial_data in subject_data]
trial_accuracies_child_same = [np.mean(trial_data == 0) for subject_data in MLEs_child_same for trial_data in subject_data]
trial_accuracies_child_diff = [np.mean(trial_data == 0) for subject_data in MLEs_child_diff for trial_data in subject_data]
# Calculate mean accuracy across trials
accuracy_adult_same = np.mean(trial_accuracies_adult_same)
accuracy_adult_diff = np.mean(trial_accuracies_adult_diff)
accuracy_child_same = np.mean(trial_accuracies_child_same)
accuracy_child_diff = np.mean(trial_accuracies_child_diff)
# Calculate 95% confidence radius around each mean accuracy
CI_adult_same = 1.96 * np.std(trial_accuracies_adult_same) / np.sqrt(len(trial_accuracies_adult_same))
CI_adult_diff = 1.96 * np.std(trial_accuracies_adult_diff) / np.sqrt(len(trial_accuracies_adult_diff))
CI_child_same = 1.96 * np.std(trial_accuracies_child_same) / np.sqrt(len(trial_accuracies_child_same))
CI_child_diff = 1.96 * np.std(trial_accuracies_child_diff) / np.sqrt(len(trial_accuracies_child_diff))
print '\nProportion of time (total frames) on target:'
print 'Adult Same: ' + str(accuracy_adult_same) + ' +/- ' + str(CI_adult_same)
print 'Adult Diff: ' + str(accuracy_adult_diff) + ' +/- ' + str(CI_adult_diff)
print 'Child Same: ' + str(accuracy_child_same) + ' +/- ' + str(CI_child_same)
print 'Child Diff: ' + str(accuracy_child_diff) + ' +/- ' + str(CI_child_diff)

# Repeat above block, but using only frames with non-missing data
# Calculate TrackIt performance (proportion of non-missing frames on target) for each trial
trial_accuracies_adult_same = [np.mean(trial_data[trial_data > -1] == 0) for subject_data in MLEs_adult_same for trial_data in subject_data]
trial_accuracies_adult_diff = [np.mean(trial_data[trial_data > -1] == 0) for subject_data in MLEs_adult_diff for trial_data in subject_data]
trial_accuracies_child_same = [np.mean(trial_data[trial_data > -1] == 0) for subject_data in MLEs_child_same for trial_data in subject_data]
trial_accuracies_child_diff = [np.mean(trial_data[trial_data > -1] == 0) for subject_data in MLEs_child_diff for trial_data in subject_data]
# Calculate mean accuracy across trials
accuracy_adult_same = np.nanmean(trial_accuracies_adult_same)
accuracy_adult_diff = np.nanmean(trial_accuracies_adult_diff)
accuracy_child_same = np.nanmean(trial_accuracies_child_same)
accuracy_child_diff = np.nanmean(trial_accuracies_child_diff)
# Calculate 95% confidence radius around each mean accuracy
CI_adult_same = 1.96 * np.nanstd(trial_accuracies_adult_same) / np.sqrt(len(trial_accuracies_adult_same))
CI_adult_diff = 1.96 * np.nanstd(trial_accuracies_adult_diff) / np.sqrt(len(trial_accuracies_adult_diff))
CI_child_same = 1.96 * np.nanstd(trial_accuracies_child_same) / np.sqrt(len(trial_accuracies_child_same))
CI_child_diff = 1.96 * np.nanstd(trial_accuracies_child_diff) / np.sqrt(len(trial_accuracies_child_diff))
print '\nProportion of time (non-missing frames) on target:'
print 'Adult Same: ' + str(accuracy_adult_same) + ' +/- ' + str(CI_adult_same)
print 'Adult Diff: ' + str(accuracy_adult_diff) + ' +/- ' + str(CI_adult_diff)
print 'Child Same: ' + str(accuracy_child_same) + ' +/- ' + str(CI_child_same)
print 'Child Diff: ' + str(accuracy_child_diff) + ' +/- ' + str(CI_child_diff)

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
