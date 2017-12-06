from load_data import load_full_subject_data
from eyetracking_hmm import getTrackItMLE
import matplotlib.pyplot as plt

# CODE FOR LOADING DATA
root = "/home/sss1/Desktop/projects/eyetracking/data/" # Office desktop
# root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
# root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
subject_type = "adult_pilot/" # "3yo/" 
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "AnnaSame.csv" # "shashank.csv" # "A232Same.csv" 
ET_fname = "AnnaSame_9_13_2016_13_25.csv" # "shashank1_12_5_2017_13_41.csv" # "A232Same_3_29_2016_10_26.csv" 

TI_file_path = root + subject_type + TI_data_dir + TI_fname
ET_file_path = root + subject_type + ET_data_dir + ET_fname

print 'Track-It file: ' + TI_file_path

eyetrack_all_trials, target_all_trials, distractors_all_trials \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

MLEs = [getTrackItMLE(eyetrack_all_trials[trial_idx], target_all_trials[trial_idx], distractors_all_trials[trial_idx]) \
        for trial_idx in range(len(target_all_trials))]
print min([len(MLEs[trial_idx]) for trial_idx in range(len(target_all_trials))])

for trial_idx in range(len(target_all_trials)):
  plt.plot(MLEs[trial_idx])
plt.show()
