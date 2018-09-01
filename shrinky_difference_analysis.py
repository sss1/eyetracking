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
from scipy import stats

# # # load datasets
# data_shrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_shrinky, dp.eyetrack_fnames_shrinky)]
# data_noshrinky = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_noshrinky, dp.eyetrack_fnames_noshrinky)]
# ages_shrinky, ages_noshrinky = read_ages() # TODO: Pass in different age files rather than hard_coding in read_ages()
# ages_tmp = [(subject_age_shrinky + subject_age_noshrinky)/2 for (subject_age_shrinky, subject_age_noshrinky) in zip(ages_shrinky, ages_noshrinky)]
# print 'Before discarding any subjects...'
# print "# of 3yo's:" + str(np.sum([3 < age and age < 4 for age in ages_tmp]))
# print "# of 4yo's:" + str(np.sum([4 < age and age < 5 for age in ages_tmp]))
# print "# of 5yo's:" + str(np.sum([5 < age and age < 6 for age in ages_tmp]))
# 
# print '\nMissing data before interpolation and discarding:'
# print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
# print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))
# 
# # Preprocess data (synchronize TrackIt with eyetracking, and interpolate some missing data, and discard trials/subjects
# # with too much missing data
# trial_discard_threshold = 0.5 # 0 discards trials with *any* missing data; 1 keeps all trials
# subject_discard_threshold = 0.5 # 0 discards subjects with *any* missing trials; 1 keeps all subjects
# data_shrinky = [preprocess_all(*subject_data, \
#                                trial_discard_threshold = trial_discard_threshold, \
#                                subject_discard_threshold = subject_discard_threshold) \
#                 for subject_data in data_shrinky]
# data_noshrinky = [preprocess_all(*subject_data, \
#                                trial_discard_threshold = trial_discard_threshold, \
#                                subject_discard_threshold = subject_discard_threshold) \
#                 for subject_data in data_noshrinky]
# 
# # Remove subjects who have too much missing data in EITHER condition, and average ages between two conditions
# assert len(data_shrinky) == len(ages_shrinky)
# assert len(ages_shrinky) == len(data_noshrinky)
# assert len(data_noshrinky) == len(ages_noshrinky)
# zipped_data = zip(data_shrinky, ages_shrinky, data_noshrinky, ages_noshrinky)
# 
# for (subject_data_shrinky, subject_age_shrinky, subject_data_noshrinky, subject_age_noshrinky) in zipped_data:
#   if subject_data_shrinky[0] is None or subject_data_noshrinky[0] is None:
#     print 'Discarding subject of average age ' + str((subject_age_shrinky + subject_age_noshrinky)/2)
# 
# zipped_data = [(subject_data_shrinky, subject_data_noshrinky, (subject_age_shrinky + subject_age_noshrinky)/2) \
#                for (subject_data_shrinky, subject_age_shrinky, subject_data_noshrinky, subject_age_noshrinky) \
#                in zipped_data if (subject_data_shrinky[0] is not None and subject_data_noshrinky[0] is not None)]
# data_shrinky, data_noshrinky, ages = zip(*zipped_data)
# 
# print '\nMissing data after interpolation and discarding:'
# print 'Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_shrinky for trial_data in subject_data[0] for x in trial_data[0]])))
# print 'No Shrinky: ' + str(np.mean(np.isnan([x for subject_data in data_noshrinky for trial_data in subject_data[0] for x in trial_data[0]])))
# 
# # Values taken from supervised results in CogSci 18 paper
# sigma2_child = 870 ** 2
# 
# # Apply HMM analysis to each dataset
# # Each MLEs_*_* is a subject X trial X timepoint array of states, taking integer values in [0, num_distractors]
# MLEs_shrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_child) for trial_data in zip(*subject_data)] for subject_data in data_shrinky]
# MLEs_noshrinky = [[eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2_child) for trial_data in zip(*subject_data)] for subject_data in data_noshrinky]
# # Since HMM analysis is slow, cache the classifications (estimating state sequences)
# np.savez(dp.root + 'cache/' + 'MLEs_shrinky_difference_analysis.npz', \
#          ages = ages, \
#          MLEs_shrinky = MLEs_shrinky, \
#          MLEs_noshrinky = MLEs_noshrinky)

# Load HMM analysis classifications from cache file
MLEs_cache_file = np.load(dp.root + 'cache/' + 'MLEs_shrinky_difference_analysis.npz')
ages = MLEs_cache_file['ages']
MLEs_shrinky = MLEs_cache_file['MLEs_shrinky']
MLEs_noshrinky = MLEs_cache_file['MLEs_noshrinky']

print 'After discarding some subjects...'
print "# of 3yo's:" + str(np.sum([3 < age and age < 4 for age in ages]))
print "# of 4yo's:" + str(np.sum([4 < age and age < 5 for age in ages]))
print "# of 5yo's:" + str(np.sum([5 < age and age < 6 for age in ages]))

# # Plot of subject performance (out of all frames) over age
# plt.figure(4)
# performance_shrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
# SEs_shrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]
# performance_noshrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
# SEs_noshrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_noshrinky]
# plt.scatter(ages_shrinky, performance_shrinky)
# plt.scatter(ages_noshrinky, performance_noshrinky)
# plt.errorbar(ages_shrinky, performance_shrinky, yerr = SEs_shrinky, linestyle = '')
# plt.errorbar(ages_noshrinky, performance_noshrinky, yerr = SEs_noshrinky, linestyle = '')
# 
# plt.legend(['Shrinky', 'No Shrinky'])
# plt.xlabel('Age (Years)')
# plt.ylabel('Proportion of all frames on target')
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
# # # Nonparametric (Nadaraya-Watson) regression
# # plt.gca().set_color_cycle(None)
# # grid = np.r_[3:6:512j]
# # # SHRINKY condition
# # k0 = smooth.NonParamRegression(X_shrinky[:,0], y_shrinky, method=npr_methods.SpatialAverage())
# # k0.fit()
# # plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
# # def fit(xs, ys):
# #   est = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
# #   est.fit()
# #   return est
# # result = bs.bootstrap(fit, X_shrinky[:,0], y_shrinky, eval_points = grid, CI = (95,99), repeats = 1000)
# # plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# # plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# # plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
# # # NOSHRINKY condition
# # k0 = smooth.NonParamRegression(X_noshrinky[:,0], y_noshrinky, method=npr_methods.SpatialAverage())
# # k0.fit()
# # plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
# # result = bs.bootstrap(fit, X_noshrinky[:,0], y_noshrinky, eval_points = grid, CI = (95,99), repeats = 1000)
# # plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# # plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# # plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
# plt.ylim((0,1))

# Plot of difference (between SHRINKY and NOSHRINKY) in subject performance (out of all frames) over age
plt.figure(1)
performance_shrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
performance_noshrinky = np.asarray([np.nanmean([float(x == 0) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
diff_performance = performance_shrinky - performance_noshrinky # np.asarray([perf_shrinky - perf_noshrinky for (perf_shrinky, perf_noshrinky) in zip(performance_shrinky, performance_noshrinky)])
print 'diff_performance.shape: ' + str(diff_performance.shape)
print 'np.nanmean(diff_performance): ' + str(np.nanmean(diff_performance))
print 'np.nanstd(diff_performance): ' + str(np.nanstd(diff_performance))
print 'p-value from paired t-test of difference in condition means ' + str(stats.ttest_1samp(diff_performance, 0))
plt.scatter(ages, diff_performance)
# SEs_shrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]

regr = linear_model.LinearRegression()
regr.fit(ages[:, np.newaxis], diff_performance)
diff_predicted = regr.predict(ages[:, np.newaxis])
plt.plot(ages[:, np.newaxis], diff_predicted)
print 'R^2 score (all frames): ' + str(r2_score(diff_performance, diff_predicted))

# Plot of difference (between SHRINKY and NOSHRINKY) in subject performance (out of good frames) over age
plt.figure(2)
def on_target_or_nan(x): # replace missing data (encoded as -1) with nan
  if x == -1:
    return float("nan")
  return x == 0
performance_shrinky = np.asarray([np.nanmean([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_shrinky])
performance_noshrinky = np.asarray([np.nanmean([on_target_or_nan(x) for trial_data in subject_data for x in trial_data]) for subject_data in MLEs_noshrinky])
diff_performance = performance_shrinky - performance_noshrinky # np.asarray([perf_shrinky - perf_noshrinky for (perf_shrinky, perf_noshrinky) in zip(performance_shrinky, performance_noshrinky)])
plt.scatter(ages, diff_performance)
# SEs_shrinky = [np.asarray(np.nanstd([float(x == 0) for trial_data in subject_data for x in trial_data]))/math.sqrt(len(subject_data)) for subject_data in MLEs_shrinky]

regr = linear_model.LinearRegression()
regr.fit(ages[:, np.newaxis], diff_performance)
diff_predicted = regr.predict(ages[:, np.newaxis])
plt.plot(ages[:, np.newaxis], diff_predicted)
print 'R^2 score (good frames only): ' + str(r2_score(diff_performance, diff_predicted))

plt.show()
