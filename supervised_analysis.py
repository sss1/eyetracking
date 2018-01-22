import numpy as np
np.set_printoptions(threshold = np.nan)
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
import sys

from load_data import load_full_subject_data
import data_paths as dp
import naive_eyetracking
import eyetracking_hmm
from util import preprocess_all

adult_cachefile = dp.root + 'cache/' + 'adult_supervised_analysis.csv'
child_cachefile = dp.root + 'cache/' + 'child_supervised_analysis.csv'

# Load and preprocess data
data_child_super = [load_full_subject_data(*entry, filter_threshold = 1, is_supervised = True) for entry in zip(dp.trackit_fnames_child_supervised, dp.eyetrack_fnames_child_supervised)]
data_adult_super = [load_full_subject_data(*entry, filter_threshold = 1, is_supervised = True) for entry in zip(dp.trackit_fnames_adult_supervised, dp.eyetrack_fnames_adult_supervised)]

data_adult_super = data_adult_super[0:1] # TODO: Remove this line! Debugging only!

print '\nMissing data before interpolation:'
print 'Child: ' + str(np.mean(np.isnan([x for subject_data in data_child_super for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult: ' + str(np.mean(np.isnan([x for subject_data in data_adult_super for trial_data in subject_data[0] for x in trial_data[0]])))

# We assume the number of distractors is the same in all trials and use this to calculate chance accuracy
num_distractors = data_adult_super[0][2][0].shape[0]
chance_accuracy = 1.0/(num_distractors + 1)
print '\nSince there are ' + str(num_distractors) + ' distractors, chance accuracy is ' + str(chance_accuracy) + '.'

# Preprocess data
data_child_super = [preprocess_all(*subject_data) for subject_data in data_child_super]
data_adult_super = [preprocess_all(*subject_data) for subject_data in data_adult_super]

# Split off true labels from unsupervised data
child_labels = [subject_data[3] for subject_data in data_child_super]
data_child_super = [subject_data[0:3] for subject_data in data_child_super]
adult_labels = [subject_data[3] for subject_data in data_adult_super]
data_adult_super = [subject_data[0:3] for subject_data in data_adult_super]

print '\nMissing data after interpolation:'
print 'Child: ' + str(np.mean(np.isnan([x for subject_data in data_child_super for trial_data in subject_data[0] for x in trial_data[0]])))
print 'Adult: ' + str(np.mean(np.isnan([x for subject_data in data_adult_super for trial_data in subject_data[0] for x in trial_data[0]])))
sys.stdout.flush()

# # For a range of thresholds between 0 and 1, plot histogram of trials per subject with >= threshold proportion of valid data
# plt.figure(1)
# data = data_child_super
# condition_name = 'Supervised'
# num_plots = 10
# x_max = max([len(subject_data[0]) for subject_data in data]) + 1
# for threshold_idx in range(num_plots):
#   threshold = threshold_idx / float(num_plots)
#   plt.subplot(num_plots, 1, threshold_idx + 1)
#   plt.hist([sum([np.mean(np.isnan(trial_data[0,:])) > threshold for trial_data in subject_data[0]]) for subject_data in data], bins = [x/2.0 for x in range(2*x_max)])
#   plt.xlim(0, x_max)
#   plt.ylim(0, len(data_child_super))
#   plt.ylabel(str(threshold))
# plt.gcf().text(0.04, 0.5, 'Threshold', va = 'center', rotation = 'vertical')
# plt.gcf().suptitle('Distribution of Trials per Subject\nwith Missing Data Proportion at most \"Thresholding\"\n(' + condition_name + ' Condition)', fontsize = "x-large")
# mng = plt.get_current_fig_manager()
# # mng.resize(*mng.window.maxsize())
# plt.show()

# Flatten data into trials (i.e., across subjects)
flattened_child_data = [trial_data for subject_data in data_child_super for trial_data in zip(*subject_data)]
flattened_adult_data = [trial_data for subject_data in data_adult_super for trial_data in zip(*subject_data)]

# Flatten labels into frames (i.e., across subject)
flattened_child_labels = [np.array(trial_data, dtype = int) for subject_data in child_labels for trial_data in subject_data]
flattened_adult_labels = [np.array(trial_data, dtype = int) for subject_data in adult_labels for trial_data in subject_data]

# Range of variance values to try
sigma2s = np.logspace(2, 8, num = 49)
# Apply HMM model to adult data at each sigma2 value
with open(adult_cachefile, 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter = ',')
  for sigma2 in sigma2s:
    MLEs_adult_super = [eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2) for trial_data in flattened_adult_data]
  
    # Calculate accuracy for each trial
    trial_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_adult_super, flattened_adult_labels)]
    adult_accuracy = np.mean(trial_accuracies)
    adult_accuracy_std_err = np.sqrt(adult_accuracy*(1 - adult_accuracy)/len(trial_accuracies))
    writer.writerow([sigma2, adult_accuracy, adult_accuracy_std_err])
    print str(sigma2) + ', ' + str(adult_accuracy) + ', ' + str(adult_accuracy_std_err)
    sys.stdout.flush()

# Apply HMM model to child data at each sigma2 value
with open(child_cachefile, 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter = ',')
  for sigma2 in sigma2s:
    MLEs_child_super = [eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2) for trial_data in flattened_child_data]
  
    # Calculate accuracy for each trial
    trial_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_child_super, flattened_child_labels)]
    child_accuracy = np.mean(trial_accuracies)
    child_accuracy_std_err = np.sqrt(child_accuracy*(1 - child_accuracy)/len(trial_accuracies))
    writer.writerow([sigma2, child_accuracy, child_accuracy_std_err])
    print str(sigma2) + ', ' + str(child_accuracy) + ', ' + str(child_accuracy_std_err)
    sys.stdout.flush()

# Apply naive model to adults
MLEs_adult_super = [naive_eyetracking.get_trackit_MLE(*trial_data) for trial_data in flattened_adult_data]
adult_naive_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_adult_super, flattened_adult_labels)]
adult_naive_accuracy = np.mean(adult_naive_accuracies)
adult_naive_accuracy_std_err = np.sqrt(adult_naive_accuracy*(1 - adult_naive_accuracy) / len(adult_naive_accuracies))
print '\nNaive model accuracy on adults is ' + str(adult_naive_accuracy) + \
      ', with standard error ' + str(adult_naive_accuracy_std_err) + \
      ', based on ' + str(len(adult_naive_accuracies)) + ' trials.'

# Apply naive model to children
MLEs_child_super = [naive_eyetracking.get_trackit_MLE(*trial_data) for trial_data in flattened_child_data]
child_naive_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_child_super, flattened_child_labels)]
child_naive_accuracy = np.mean(child_naive_accuracies)
child_naive_accuracy_std_err = np.sqrt(child_naive_accuracy*(1 - child_naive_accuracy) / len(child_naive_accuracies))
print '\nNaive model accuracy on children is ' + str(child_naive_accuracy) + \
      ', with standard error ' + str(child_naive_accuracy_std_err) + \
      ', based on ' + str(len(child_naive_accuracies)) + ' trials.'

# Read results from adult_cachefile
accuracies = np.zeros(sigma2s.shape)
std_errs = np.zeros(sigma2s.shape)
with open(adult_cachefile, 'rb') as csvfile:
  reader = csv.reader(csvfile, delimiter = ',')
  row_idx = 0
  for row in reader:
    accuracies[row_idx] = float(row[1])
    std_errs[row_idx] = float(row[2])
    row_idx += 1

# Plot model accuracy as a function of sigma2
type1_error = 0.05
bonferroni_type1_error = type1_error/len(std_errs) # Convert error bars to uniform error bands via Bonferroni
yerr = norm.ppf(1 - bonferroni_type1_error/2) * std_errs # Radius of confidence band
acc_line, = plt.plot(np.sqrt(sigma2s), accuracies, c = 'b', marker = 'o', zorder = 2, label = 'HMM (Adult)') # point estimates
lower_band, = plt.plot(np.sqrt(sigma2s), accuracies - yerr, c = 'b', ls = '--', zorder = 2) # upper confidence band
upper_band, = plt.plot(np.sqrt(sigma2s), accuracies + yerr, c = 'b', ls = '--', zorder = 2) # lower confidence band

min_max_x = np.sqrt([sigma2s[0], sigma2s[-1]])
# Plot accuracy for guessing the closest object
adult_naive_val = np.array([adult_naive_accuracy, adult_naive_accuracy])
adult_naive_yerr = norm.ppf(1 - type1_error/2) * adult_naive_accuracy_std_err
adult_naive_line, = plt.plot(min_max_x, adult_naive_val, c = 'g', label = 'Naive (Adult)')
adult_naive_lower_band, = plt.plot(min_max_x, adult_naive_val - adult_naive_yerr, c = 'g', ls = '--', zorder = 2) # upper confidence band
adult_naive_upper_band, = plt.plot(min_max_x, adult_naive_val + adult_naive_yerr, c = 'g', ls = '--', zorder = 2) # lower confidence band
# Plot accuracy for random guessing
chance_line, = plt.plot(min_max_x, [chance_accuracy, chance_accuracy], c = 'r', label = 'Chance')

plt.xlabel('$\sigma$ (pixels)', fontsize = 24)
plt.ylabel('Classification Accuracy', fontsize = 24)
plt.gca().set_xscale("log", nonposx = 'clip')
plt.xlim(min_max_x)
plt.ylim((0, 1))
min_idx = np.argmax(accuracies)
opt_point = plt.scatter(np.sqrt(sigma2s[min_idx]), accuracies[min_idx], c = 'r', marker = '^', s = 100, zorder = 3, label = 'Optimum')
plt.legend(handles = [acc_line, adult_naive_line, chance_line, opt_point], numpoints = 3, scatterpoints = 1, loc = 'lower left')
plt.show()
