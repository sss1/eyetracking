import numpy as np
import matplotlib.pyplot as plt

from load_data import load_full_subject_data
import data_paths as dp
from eyetracking_hmm import get_trackit_MLE
from util import preprocess_all

data_child_super = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_supervised, dp.eyetrack_fnames_child_supervised)]

# TODO: Need to rewrite preprocess_all to handle labeled data
data_child_super = [preprocess_all(*subject_data) for subject_data in data_child_super]

# Split off true labels from unsupervised data
labels = [subject_data[3] for subject_data in data_child_super]
data_child_super = [subject_data[0:3] for subject_data in data_child_super]

# For a range of thresholds between 0 and 1, plot histogram of trials per subject with >= threshold proportion of valid data
plt.figure(1)
data = data_child_super
condition_name = 'Supervised'
num_plots = 10
x_max = max([len(subject_data[0]) for subject_data in data]) + 1
for threshold_idx in range(num_plots):
  threshold = threshold_idx / float(num_plots)
  plt.subplot(num_plots, 1, threshold_idx + 1)
  plt.hist([sum([np.mean(np.isnan(trial_data[0,:])) > threshold for trial_data in subject_data[0]]) for subject_data in data], bins = [x/2.0 for x in range(2*x_max)])
  plt.xlim(0, x_max)
  plt.ylim(0, len(data_child_super))
  plt.ylabel(str(threshold))
plt.gcf().text(0.04, 0.5, 'Threshold', va = 'center', rotation = 'vertical')
plt.gcf().suptitle('Distribution of Trials per Subject\nwith Missing Data Proportion at most \"Thresholding\"\n(' + condition_name + ' Condition)', fontsize = "x-large")
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

# Flatten data into trials (i.e., across subjects)
flattened_data = [trial_data for subject_data in data_child_super for trial_data in subject_data]

# Flatten labels into frames (i.e., across subject and trials)
labels = [frame_data for subject_data in labels for trial_data in subject_data for frame_data in trial_data]

# Range of variance values to try
sigma2s = np.logspace(1, 4, num = 10)

for sigma2 in sigma2s:
  MLEs_child_super = [get_trackit_MLE(*trial_data, sigma2 = sigma2) for trial_data in flattened]
  MLEs_flattened = [frame_data for trial_data in MLEs_child_super for frame_data in trial_data] # flatten MLEs across trials
  print 'For sigma2 = ' + str(sigma2) + ', the error is ' + str(compute(MLEs_flattened, labels)) + '.'

def compute_estimation_error(estimated_states, true_states):
  return np.array([compute_trial_estimation_error(*trial_data) for trial_data in zip(estimated_states, true_states)]).mean()

def compute_trial_estimation_error(estimated_trial_states, true_trial_states):
  return (estimated_trial_states != true_trial_states).mean()
