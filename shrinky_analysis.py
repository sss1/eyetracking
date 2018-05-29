from load_data import load_full_subject_data
import eyetracking_hmm
import naive_eyetracking
import matplotlib.pyplot as plt
from util import preprocess_all, jagged_to_numpy
import data_paths_full as dp # Large unsupervised (shrinky) dataset
import numpy as np
from read_ages import read_ages

# load datasets
data_shrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_shrinky, dp.eyetrack_fnames_shrinky)]
data_noshrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_noshrinky, dp.eyetrack_fnames_noshrinky)]

print '\nMissing data before interpolation and discarding:'
print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))

# Preprocess data (synchronize TrackIt with eyetracking, and interpolate some missing data, and discard trials/subjects
# with too much missing data
trial_discard_threshold = 0.5 # 0 discards trials with *any* missing data; 1 keeps all trials
subject_discard_threshold = 1 # 0 discards subjects with *any* missing trials; 1 keeps all subjects
data_shrinky = [preprocess_all(*subject_data, \
                               trial_discard_threshold = trial_discard_threshold, \
                               subject_discard_threshold = subject_discard_threshold) \
                for subject_data in data_shrinky]
data_shrinky = [subject_data for subject_data in data_shrinky if subject_data[0] is not None]
data_noshrinky = [preprocess_all(*subject_data, \
                               trial_discard_threshold = trial_discard_threshold, \
                               subject_discard_threshold = subject_discard_threshold) \
                for subject_data in data_noshrinky]
data_noshrinky = [subject_data for subject_data in data_noshrinky if subject_data[0] is not None]

print '\nMissing data after interpolation and discarding:'
print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))

# Values taken from supervised results in CogSci 18 paper
sigma2_adult = 490 ** 2
sigma2_child = 870 ** 2

# # Apply HMM analysis to each dataset
# # Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
# MLEs_shrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_adult) for trial_data in zip(*subject_data)] for subject_data in data_shrinky]
# MLEs_noshrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_adult) for trial_data in zip(*subject_data)] for subject_data in data_noshrinky]
# # Since HMM analysis is slow, cache the classifications (estimating state sequences)
# np.savez(dp.root + 'cache/' + 'MLEs_shrinky_analysis.npz', \
#          MLEs_shrinky = MLEs_shrinky, \
#          MLEs_noshrinky = MLEs_noshrinky)

# Load HMM analysis classifications from cache file
MLEs_cache_file = np.load(dp.root + 'cache/' + 'MLEs_shrinky_analysis.npz')
MLEs_shrinky = MLEs_cache_file['MLEs_shrinky']
MLEs_noshrinky = MLEs_cache_file['MLEs_noshrinky']

# Plot proportion of (all) trials on target over trial time
plt.figure(1)
aligned_shrinky = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
aligned_noshrinky = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
plt.plot(np.nanmean(aligned_shrinky, axis = 0))
plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
plt.legend(['Shrinky', 'No Shrinky'])
# For each dataset, plot a vertical line indicating the length of the shortest trial
min_trial_len_shrinky = min([len(trial_data) for subject_data in MLEs_shrinky for trial_data in subject_data])
min_trial_len_noshrinky = min([len(trial_data) for subject_data in MLEs_noshrinky for trial_data in subject_data])
plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
plt.xlabel('Time (frames, at 60Hz)')
plt.ylabel('Fraction of all trials on target')

# Plot proportion of (non-missing or interpolated) trials on target over trial time
plt.figure(2)
def on_target_or_nan(x): # replace missing data (encoded as -1) with nan
  if x == -1:
    return float("nan")
  return x == 0
aligned_shrinky = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
aligned_noshrinky = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
plt.plot(np.nanmean(aligned_shrinky, axis = 0))
plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
plt.legend(['Shrinky', 'No Shrinky'])
plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
plt.xlabel('Time (frames, at 60Hz)')
plt.ylabel('Fraction of non-missing or interpolated frames on target')

# Plot proportion of missing data over trial time
plt.figure(3)
aligned_shrinky = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
aligned_noshrinky = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
plt.plot(np.nanmean(aligned_shrinky, axis = 0))
plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
plt.legend(['Shrinky', 'No Shrinky'])
plt.gca().set_color_cycle(None)
plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
plt.xlabel('Trial Time (frames, at 60Hz)')
plt.ylabel('Fraction of frames missing data')

# Plot of subject performance over age
plt.figure(4)
performance_shrinky = [np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky]
performance_noshrinky = [np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky]
shrinky_ages, noshrinky_ages = read_ages()
plt.scatter(shrinky_ages, performance_shrinky)
plt.scatter(noshrinky_ages, performance_noshrinky)
plt.xlabel('Age (Years)')
plt.ylabel('Fraction of all frames on target')
plt.show()
