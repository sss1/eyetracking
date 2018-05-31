from load_data import load_full_subject_data
import eyetracking_hmm
import naive_eyetracking
import matplotlib.pyplot as plt
from util import preprocess_all, jagged_to_numpy
import data_paths_full as dp # Large unsupervised (shrinky) dataset
import numpy as np
from read_ages import read_ages
import math
from sklearn import linear_model
from sklearn.metrics import r2_score
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
import pyqt_fit.bootstrap as bs

# load datasets
data_shrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_shrinky, dp.eyetrack_fnames_shrinky)]
data_noshrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_noshrinky, dp.eyetrack_fnames_noshrinky)]
ages_shrinky, ages_noshrinky = read_ages()

print '\nMissing data before interpolation and discarding:'
print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))

# Preprocess data (synchronize TrackIt with eyetracking, and interpolate some missing data, and discard trials/subjects
# with too much missing data
trial_discard_threshold = 0.5 # 0 discards trials with *any* missing data; 1 keeps all trials
subject_discard_threshold = 0.5 # 0 discards subjects with *any* missing trials; 1 keeps all subjects
data_shrinky = [preprocess_all(*subject_data, \
                               trial_discard_threshold = trial_discard_threshold, \
                               subject_discard_threshold = subject_discard_threshold) \
                for subject_data in data_shrinky]
ages_shrinky = [age for (age,subject_data) in zip(ages_shrinky, data_shrinky) if subject_data[0] is not None]
print ages_shrinky
data_shrinky = [subject_data for subject_data in data_shrinky if subject_data[0] is not None]
data_noshrinky = [preprocess_all(*subject_data, \
                               trial_discard_threshold = trial_discard_threshold, \
                               subject_discard_threshold = subject_discard_threshold) \
                for subject_data in data_noshrinky]
ages_noshrinky = [age for (age,subject_data) in zip(ages_noshrinky, data_noshrinky) if subject_data[0] is not None]
print ages_noshrinky
data_noshrinky = [subject_data for subject_data in data_noshrinky if subject_data[0] is not None]

print '\nMissing data after interpolation and discarding:'
print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))

# Values taken from supervised results in CogSci 18 paper
sigma2_child = 870 ** 2

# Apply HMM analysis to each dataset
# Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
MLEs_shrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_child) for trial_data in zip(*subject_data)] for subject_data in data_shrinky]
MLEs_noshrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_adult) for trial_data in zip(*subject_data)] for subject_data in data_noshrinky]
# Since HMM analysis is slow, cache the classifications (estimating state sequences)
np.savez(dp.root + 'cache/' + 'MLEs_shrinky_analysis.npz', \
         MLEs_shrinky = MLEs_shrinky, \
         MLEs_noshrinky = MLEs_noshrinky)

# Load HMM analysis classifications from cache file
MLEs_cache_file = np.load(dp.root + 'cache/' + 'MLEs_shrinky_analysis.npz')
MLEs_shrinky = MLEs_cache_file['MLEs_shrinky']
MLEs_noshrinky = MLEs_cache_file['MLEs_noshrinky']

# # Plot proportion of (all) trials on target over trial time
# plt.figure(1)
# aligned_shrinky = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
# aligned_noshrinky = jagged_to_numpy([[float(x == 0) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
# plt.plot(np.nanmean(aligned_shrinky, axis = 0))
# plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
# plt.legend(['Shrinky', 'No Shrinky'])
# # For each dataset, plot a vertical line indicating the length of the shortest trial
# min_trial_len_shrinky = min([len(trial_data) for subject_data in MLEs_shrinky for trial_data in subject_data])
# min_trial_len_noshrinky = min([len(trial_data) for subject_data in MLEs_noshrinky for trial_data in subject_data])
# plt.gca().set_color_cycle(None)
# plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
# plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
# plt.xlabel('Time (frames, at 60Hz)')
# plt.ylabel('Fraction of all trials on target')
# 
# # Plot proportion of (non-missing or interpolated) trials on target over trial time
# plt.figure(2)
# def on_target_or_nan(x): # replace missing data (encoded as -1) with nan
#   if x == -1:
#     return float("nan")
#   return x == 0
# aligned_shrinky = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
# aligned_noshrinky = jagged_to_numpy([[on_target_or_nan(x) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
# plt.plot(np.nanmean(aligned_shrinky, axis = 0))
# plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
# plt.legend(['Shrinky', 'No Shrinky'])
# plt.gca().set_color_cycle(None)
# plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
# plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
# plt.xlabel('Time (frames, at 60Hz)')
# plt.ylabel('Fraction of non-missing or interpolated frames on target')
# 
# # Plot proportion of missing data over trial time
# plt.figure(3)
# aligned_shrinky = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_shrinky for trial_data in subject_data])
# aligned_noshrinky = jagged_to_numpy([[float(x == -1) for x in trial_data] for subject_data in MLEs_noshrinky for trial_data in subject_data])
# plt.plot(np.nanmean(aligned_shrinky, axis = 0))
# plt.plot(np.nanmean(aligned_noshrinky, axis = 0))
# plt.legend(['Shrinky', 'No Shrinky'])
# plt.gca().set_color_cycle(None)
# plt.plot([min_trial_len_shrinky, min_trial_len_shrinky], [0, 1])
# plt.plot([min_trial_len_noshrinky, min_trial_len_noshrinky], [0, 1])
# plt.xlabel('Trial Time (frames, at 60Hz)')
# plt.ylabel('Fraction of frames missing data')

# Plot of subject performance (out of all frames) over age
plt.figure(4)
performance_shrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
SEs_shrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]
performance_noshrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
SEs_noshrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_noshrinky]
plt.scatter(ages_shrinky, performance_shrinky)
plt.scatter(ages_noshrinky, performance_noshrinky)
plt.errorbar(ages_shrinky, performance_shrinky, yerr = SEs_shrinky, linestyle = '')
plt.errorbar(ages_noshrinky, performance_noshrinky, yerr = SEs_noshrinky, linestyle = '')

plt.legend(['Shrinky', 'No Shrinky'])
plt.xlabel('Age (Years)')
plt.ylabel('Proportion of all frames on target')

# Plot regression lines
plt.gca().set_color_cycle(None)
# SHRINKY condition
X_shrinky = ages_shrinky[~np.isnan(performance_shrinky), np.newaxis]
y_shrinky = performance_shrinky[~np.isnan(performance_shrinky)]
regr_shrinky = linear_model.LinearRegression()
regr_shrinky.fit(X_shrinky, y_shrinky)
performance_pred_shrinky = regr_shrinky.predict(X_shrinky)
plt.plot(X_shrinky, performance_pred_shrinky)
# NOSHRINKY condition
X_noshrinky = ages_noshrinky[~np.isnan(performance_noshrinky), np.newaxis]
y_noshrinky = performance_noshrinky[~np.isnan(performance_noshrinky)]
regr_noshrinky = linear_model.LinearRegression()
regr_noshrinky.fit(X_noshrinky, y_noshrinky)
performance_pred_noshrinky = regr_noshrinky.predict(X_noshrinky)
plt.plot(X_noshrinky, performance_pred_noshrinky)
# # Nonparametric (Nadaraya-Watson) regression
# plt.gca().set_color_cycle(None)
# grid = np.r_[3:6:512j]
# # SHRINKY condition
# k0 = smooth.NonParamRegression(X_shrinky[:,0], y_shrinky, method=npr_methods.SpatialAverage())
# k0.fit()
# plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
# def fit(xs, ys):
#   est = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#   est.fit()
#   return est
# result = bs.bootstrap(fit, X_shrinky[:,0], y_shrinky, eval_points = grid, CI = (95,99), repeats = 1000)
# plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
# # NOSHRINKY condition
# k0 = smooth.NonParamRegression(X_noshrinky[:,0], y_noshrinky, method=npr_methods.SpatialAverage())
# k0.fit()
# plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
# result = bs.bootstrap(fit, X_noshrinky[:,0], y_noshrinky, eval_points = grid, CI = (95,99), repeats = 1000)
# plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
plt.ylim((0,1))

# # Plot of subject performance (out of all non-missing/interpolated) over age
# plt.figure(5)
# performance_shrinky = np.asarray([np.nanmean([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
# SEs_shrinky = [np.asarray(np.nanstd([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]
# performance_noshrinky = np.asarray([np.nanmean([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
# SEs_noshrinky = [np.asarray(np.nanstd([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_noshrinky]
# ages_shrinky, ages_noshrinky = read_ages()
# plt.scatter(ages_shrinky, performance_shrinky)
# plt.scatter(ages_noshrinky, performance_noshrinky)
# plt.errorbar(ages_shrinky, performance_shrinky, yerr = SEs_shrinky, linestyle = '')
# plt.errorbar(ages_noshrinky, performance_noshrinky, yerr = SEs_noshrinky, linestyle = '')
# 
# plt.legend(['Shrinky', 'No Shrinky'])
# plt.xlabel('Age (Years)')
# plt.ylabel('Proportion of non-missing or interpolated frames on target')
# 
# # Plot regression lines
# plt.gca().set_color_cycle(None)
# # SHRINKY condition
# X_shrinky = ages_shrinky[~np.isnan(performance_shrinky), np.newaxis]
# y_shrinky = performance_shrinky[~np.isnan(performance_shrinky)]
# regr_shrinky = linear_model.LinearRegression()
# regr_shrinky.fit(X_shrinky, y_shrinky)
# performance_pred_shrinky = regr_shrinky.predict(X_shrinky)
# plt.plot(X_shrinky, performance_pred_shrinky)
# # NOSHRINKY condition
# X_noshrinky = ages_noshrinky[~np.isnan(performance_noshrinky), np.newaxis]
# y_noshrinky = performance_noshrinky[~np.isnan(performance_noshrinky)]
# regr_noshrinky = linear_model.LinearRegression()
# regr_noshrinky.fit(X_noshrinky, y_noshrinky)
# performance_pred_noshrinky = regr_noshrinky.predict(X_noshrinky)
# plt.plot(X_noshrinky, performance_pred_noshrinky)
# 
# # Plot of missing data (proportion of all frames) over age
# plt.figure(6)
# missing_data_shrinky = np.asarray([np.nanmean([float(x == -1) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
# # SEs_shrinky = [np.asarray(np.nanstd([float(x == -1) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]
# missing_data_noshrinky = np.asarray([np.nanmean([float(x == -1) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
# # SEs_noshrinky = [np.asarray(np.nanstd([float(x == -1) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_noshrinky]
# ages_shrinky, ages_noshrinky = read_ages()
# plt.scatter(ages_shrinky, missing_data_shrinky)
# plt.scatter(ages_noshrinky, missing_data_noshrinky)
# # plt.errorbar(ages_shrinky, missing_data_shrinky, yerr = SEs_shrinky, linestyle = '')
# # plt.errorbar(ages_noshrinky, missing_data_noshrinky, yerr = SEs_noshrinky, linestyle = '')
# 
# plt.legend(['Shrinky', 'No Shrinky'])
# plt.xlabel('Age (Years)')
# plt.ylabel('Proportion of missing frames')
# 
# # Plot regression lines
# plt.gca().set_color_cycle(None)
# # SHRINKY condition
# X_shrinky = ages_shrinky[~np.isnan(missing_data_shrinky), np.newaxis]
# y_shrinky = missing_data_shrinky[~np.isnan(missing_data_shrinky)]
# regr_shrinky = linear_model.LinearRegression()
# regr_shrinky.fit(X_shrinky, y_shrinky)
# missing_data_pred_shrinky = regr_shrinky.predict(X_shrinky)
# plt.plot(X_shrinky, missing_data_pred_shrinky)
# # NOSHRINKY condition
# X_noshrinky = ages_noshrinky[~np.isnan(missing_data_noshrinky), np.newaxis]
# y_noshrinky = missing_data_noshrinky[~np.isnan(missing_data_noshrinky)]
# regr_noshrinky = linear_model.LinearRegression()
# regr_noshrinky.fit(X_noshrinky, y_noshrinky)
# missing_data_pred_noshrinky = regr_noshrinky.predict(X_noshrinky)
# plt.plot(X_noshrinky, missing_data_pred_noshrinky)
# plt.ylim((0,1))

plt.show()
