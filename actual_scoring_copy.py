# import csv
import fnmatch
# import time
# import sys
import math
import numpy as np
import os
from load_data import load_full_subject_data

# squared distance error
def error_fcn(et_x, et_y, trackit_x, trackit_y):
    return math.sqrt((et_x - trackit_x)**2 + (et_y - trackit_y)**2)

def read_and_analyze_file(maindir, subfolder, subject, filter_threshold = 1.0):
    fName = None
    dirName = maindir + "TrackItOutput/" + subfolder + "/"
    for file in os.listdir(dirName):
        if fnmatch.fnmatch(file, subject + "*.csv"):
            fName = file
            
    if fName == None:
        print 'Track-It file for subject ' + subject + ' not found.'
        return None
    TI_file_path = dirName + fName

    dirName = maindir + "EyeTracker/" + subfolder + "/"
    fName = None
    for file in os.listdir(dirName):
        if fnmatch.fnmatch(file, subject + "*.csv"):
            fName = file
    if fName == None:
        print 'Eyetracking file for subject ' + subject + ' not found.'
        return None
    ET_file_path = dirName + fName
    track_it_xy_list, distractors_xy_list, eye_track_xy_list \
      = load_full_subject_data(TI_file_path,
                                ET_file_path,
                                filter_threshold = filter_threshold)

    all_trials_error_over_time = []

    for trial_idx in range(len(track_it_xy_list)):

        # x and y lists for the current trial
        track_it_xs = track_it_xy_list[trial_idx][0]
        track_it_ys = track_it_xy_list[trial_idx][1]
        eye_track_xs = eye_track_xy_list[trial_idx][0]
        eye_track_ys = eye_track_xy_list[trial_idx][1]

        trial_length = min(2*(len(track_it_xs) - 1), len(eye_track_xs))
        trial_error_over_time = []
        for time in range(trial_length):
            trackit_time = time/2
            trackit_x = track_it_xs[trackit_time]
            trackit_y = track_it_ys[trackit_time]
            if (time % 2 != 0):
                trackit_x = (trackit_x + track_it_xs[trackit_time + 1])/2
                trackit_y = (trackit_y + track_it_ys[trackit_time + 1])/2
            trial_error_over_time.append(error_fcn(eye_track_xs[time],
                                                   eye_track_ys[time],
                                                   trackit_x,
                                                   trackit_y))
                                         
        all_trials_error_over_time.append(trial_error_over_time)

    # For each time point, the mean (across trials) error at that time
    mean_errors_by_time = map(np.mean, zip(*all_trials_error_over_time))

    return mean_errors_by_time, all_trials_error_over_time
