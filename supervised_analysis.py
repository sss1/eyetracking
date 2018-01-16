import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm

from load_data import load_full_subject_data
import data_paths as dp
from eyetracking_hmm import get_trackit_MLE
from util import preprocess_all

outfile = 'supervised_analysis.csv'

# Load and preprocess data
data_child_super = [load_full_subject_data(*entry, filter_threshold = 1, is_supervised = True) for entry in zip(dp.trackit_fnames_child_supervised, dp.eyetrack_fnames_child_supervised)]
print '\nMissing data before interpolation:'
print np.mean(np.isnan([x for subject_data in data_child_super for trial_data in subject_data[0] for x in trial_data[0]]))
data_child_super = [preprocess_all(*subject_data) for subject_data in data_child_super]


# Split off true labels from unsupervised data
labels = [subject_data[3] for subject_data in data_child_super]
data_child_super = [subject_data[0:3] for subject_data in data_child_super]

print '\nMissing data after interpolation:'
print np.mean(np.isnan([x for subject_data in data_child_super for trial_data in subject_data[0] for x in trial_data[0]]))

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

# TODO: WE CAN'T FLATTEN ACROSS TRIALS BEFORE COMPUTING CONFIDENCE INTERVALS, BECAUSE WE NEED INDEPENDENT SAMPLES!

# Flatten data into trials (i.e., across subjects)
flattened_data = [trial_data for subject_data in data_child_super for trial_data in zip(*subject_data)]

# Flatten labels into frames (i.e., across subject and trials)
flattened_labels = np.array([frame_data for subject_data in labels for trial_data in subject_data for frame_data in trial_data], dtype = int)

# Range of variance values to try
sigma2s = np.logspace(2, 8, num = 49)

# Since fitting the HMM model takes a while, cache the results in a small CSV file outfile
with open(outfile, 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter = ',')
  for sigma2 in sigma2s:
    MLEs_child_super = [get_trackit_MLE(*trial_data, sigma2 = sigma2) for trial_data in flattened_data]
    MLEs_flattened = np.array([frame_data for trial_data in MLEs_child_super for frame_data in trial_data], dtype = int) # flatten MLEs across trials
    err = (MLEs_flattened[MLEs_flattened != -1] != flattened_labels[MLEs_flattened != -1]).mean() # For now, ignore frames with missing data
    std_err = np.sqrt(err*(1 - err)/len(MLEs_flattened[MLEs_flattened != -1]))
    # print 'Couldn\'t classify ' + str((MLEs_flattened == -1).mean()) + ' fraction of frames due to missing data.'
    print str(sigma2) + ', ' + str(err) + ', ' + str(std_err)
    writer.writerow([sigma2, err, std_err])

# Read results from outfile
errors = np.zeros(sigma2s.shape)
std_errs = np.zeros(sigma2s.shape)
with open(outfile, 'rb') as csvfile:
  reader = csv.reader(csvfile, delimiter = ',')
  row_idx = 0
  for row in reader:
    errors[row_idx] = float(row[1])
    std_errs[row_idx] = float(row[2])
    row_idx += 1

type1_error = 0.05
bonferroni_type1_error = type1_error/len(std_errs) # Convert error bars to uniform error bands via Bonferroni
plt.errorbar(np.sqrt(sigma2s), errors, yerr = norm.ppf(1 - bonferroni_type1_error/2) * std_errs, zorder = 2)
plt.plot(np.sqrt(sigma2s), errors)
plt.xlabel('Standard Deviation (pixels)', fontsize = 24)
plt.ylabel('Classification Error', fontsize = 24)
ax = plt.gca()
ax.set_xscale("log", nonposx = 'clip')
plt.ylim((0, 1))
min_idx = np.argmin(errors)
plt.scatter(np.sqrt(sigma2s[min_idx]), errors[min_idx], c = 'r', s = 100, zorder = 1)
plt.show()
