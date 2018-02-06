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

def run_analysis(dataset_name, show_meta, use_all_frames):

  # Load appropriate dataset
  if dataset_name == 'Adult':
    data_source = zip(dp.trackit_fnames_adult_supervised, dp.eyetrack_fnames_adult_supervised)
  elif dataset_name == 'Child':
    data_source = zip(dp.trackit_fnames_child_supervised, dp.eyetrack_fnames_child_supervised)

  data = [load_full_subject_data(*entry, filter_threshold = 1, is_supervised = True) for entry in data_source]

  if use_all_frames:
    name = 'all_frames'
  else:
    name = 'nonmissing_only'

  print '\nRunning ' + dataset_name + ' supervised data...'
  cachefile = dp.root + 'cache/' + dataset_name.lower() + '_supervised_analysis_' + name + '.csv'

  print 'Missing data before interpolation: ' + \
    str(np.mean(np.isnan([x for subject_data in data for trial_data in subject_data[0] for x in trial_data[0]])))

  # We assume the number of distractors is the same in all trials and use this to calculate chance accuracy
  num_distractors = data[0][2][0].shape[0]
  chance_accuracy = 1.0/(num_distractors + 1)
  print 'Since there are ' + str(num_distractors) + ' distractors, chance accuracy is ' + str(chance_accuracy) + '.'
  
  # Preprocess data
  data = [preprocess_all(*subject_data) for subject_data in data]
  data = [subject_data for subject_data in data if subject_data[0] is not None]
  
  # Split off true labels from unsupervised data
  labels = [subject_data[3] for subject_data in data]
  data = [subject_data[0:3] for subject_data in data]
  
  print 'Missing data after interpolation: ' + \
    str(np.mean(np.isnan([x for subject_data in data for trial_data in subject_data[0] for x in trial_data[0]])))
  sys.stdout.flush()

  # Flatten data into trials (i.e., across subjects)
  flattened_data = [trial_data for subject_data in data for trial_data in zip(*subject_data)]
  print 'There are ' + str(len(flattened_data)) + ' remaining usable trials after preprocessing.'
  
  # Flatten labels into frames (i.e., across subject)
  flattened_labels = [np.array(trial_data, dtype = int) for subject_data in labels for trial_data in subject_data]
  
  # Range of variance values to try
  sigma2s = np.logspace(2, 8, num = 49)
  # Apply HMM model at each sigma2 value
  print 'Caching results in ' + cachefile
  with open(cachefile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    for sigma2 in sigma2s:
      MLEs_super = [eyetracking_hmm.get_trackit_MLE(*trial_data, sigma2 = sigma2) for trial_data in flattened_data]

      # Calculate accuracy for each trial
      if use_all_frames:
        trial_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_super, flattened_labels)]
      else:
        trial_accuracies = [(estimate[estimate > -1] == truth[estimate > -1]).mean() for (estimate, truth) in zip(MLEs_super, flattened_labels)]
      # Calculate mean and standard error across trials
      accuracy = np.nanmean(trial_accuracies)
      accuracy_std_err = np.nanstd(trial_accuracies) / np.sqrt(len(trial_accuracies))
      writer.writerow([sigma2, accuracy, accuracy_std_err])
      print str(sigma2) + ', ' + str(accuracy) + ', ' + str(accuracy_std_err)
      sys.stdout.flush()
  
  # Run naive model for comparison
  MLEs_super = [naive_eyetracking.get_trackit_MLE(*trial_data) for trial_data in flattened_data]
  if use_all_frames:
    naive_accuracies = [(estimate == truth).mean() for (estimate, truth) in zip(MLEs_super, flattened_labels)]
  else:
    naive_accuracies = [(estimate[estimate > -1] == truth[estimate > -1]).mean() for (estimate, truth) in zip(MLEs_super, flattened_labels)]
  naive_accuracy = np.nanmean(naive_accuracies)
  naive_accuracy_std_err = np.nanstd(naive_accuracies) / np.sqrt(len(naive_accuracies))
  print 'Naive model accuracy on is ' + str(naive_accuracy) + \
        ', with CI ' + str(1.96 * naive_accuracy_std_err) + \
        ', based on ' + str(len(naive_accuracies)) + ' trials.'
  
  # Read results from cachefile
  accuracies = np.zeros(sigma2s.shape)
  std_errs = np.zeros(sigma2s.shape)
  with open(cachefile, 'rb') as csvfile:
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
  acc_line, = plt.plot(np.sqrt(sigma2s), accuracies, c = 'b', marker = 'o', markersize = 4, zorder = 2, label = 'HMM') # point estimates
  lower_band, = plt.plot(np.sqrt(sigma2s), accuracies - yerr, c = 'b', ls = '--', zorder = 2) # upper confidence band
  upper_band, = plt.plot(np.sqrt(sigma2s), accuracies + yerr, c = 'b', ls = '--', zorder = 2) # lower confidence band
  
  min_max_x = np.sqrt([sigma2s[0], sigma2s[-1]])
  # Plot accuracy for guessing the closest object
  naive_val = np.array([naive_accuracy, naive_accuracy])
  naive_yerr = norm.ppf(1 - type1_error/2) * naive_accuracy_std_err
  naive_line, = plt.plot(min_max_x, naive_val, c = 'g', label = 'Naive')
  naive_lower_band, = plt.plot(min_max_x, naive_val - naive_yerr, c = 'g', ls = '--', zorder = 2) # upper confidence band
  naive_upper_band, = plt.plot(min_max_x, naive_val + naive_yerr, c = 'g', ls = '--', zorder = 2) # lower confidence band
  # Plot accuracy of random guessing; note that this is only accurate when using only nonmissing frames
  chance_line, = plt.plot(min_max_x, [chance_accuracy, chance_accuracy], c = 'r', label = 'Chance')
  
  plt.xlabel('$\sigma$ (pixels)', fontsize = 24)
  plt.ylabel('Classification Accuracy', fontsize = 24)
  plt.gca().set_xscale("log", nonposx = 'clip')
  plt.xlim(min_max_x)
  plt.ylim((0, 1))
  min_idx = np.argmax(accuracies)
  opt_x = np.sqrt(sigma2s[min_idx])
  opt_y = accuracies[min_idx]
  opt_point = plt.scatter(opt_x, opt_y, c = 'r', marker = '^', s = 100, zorder = 3, label = 'Optimum')
  vdiff = 0.3
  if not show_meta:
    vdiff = 0.5
  plt.annotate(str((round(opt_x, -1), round(opt_y, 2))), xy = (opt_x, opt_y), xytext = (opt_x, opt_y - vdiff), \
               arrowprops = dict(facecolor = (0, 0, 0, 0.5), edgecolor = (0, 0, 0, 0), width = 2), \
               horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 16)
  if show_meta:
    plt.legend(handles = [acc_line, naive_line, chance_line, opt_point], numpoints = 3, scatterpoints = 1, fontsize = 16)
    plt.gcf().tight_layout()
  # plt.show()
  plt.gcf().savefig('/home/sss1/Desktop/academic/projects/eyetracking/figs/' + dataset_name.lower() + '_supervised_performance_over_sigma_' + name + '.pdf')
  plt.clf()

run_analysis('Adult', show_meta = True, use_all_frames = True)
run_analysis('Child', show_meta = False, use_all_frames = True)

run_analysis('Adult', show_meta = True, use_all_frames = False)
run_analysis('Child', show_meta = False, use_all_frames = False)
