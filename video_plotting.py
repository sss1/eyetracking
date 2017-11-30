import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
from load_data import load_full_subject_data
from util import error_fcn

root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/"
subject_type = "3yo/" # "adult_pilot/"
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "A232Same.csv" # "AnnaSame.csv"
ET_fname = "A232Same_3_29_2016_10_26.csv" # "AnnaSame_9_13_2016_13_25.csv"
TI_file_path = root + subject_type + TI_data_dir + TI_fname
ET_file_path = root + subject_type + ET_data_dir + ET_fname

print 'Track-It file: ' + TI_file_path

track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

# boundaries of track-it grid
x_min = 600
x_max = 1400
y_min = 100
y_max = 900

space = 50 # number of extra pixels to display on either side of the plot

# First set up the figure, the axis, and the plot element we want to animate
lag = 10 # plot a time window of length lag, so we can see the trajectory
distractor_lag = 5
print 'Number of trials: ' + str(len(track_it_xy_list))
for trial_idx in range(len(track_it_xy_list)):

    # x and y lists for the current trial
    track_it_xs = track_it_xy_list[trial_idx][0]
    track_it_ys = track_it_xy_list[trial_idx][1]
    distractors = distractors_xy_list[trial_idx]
    eye_track_xs = eye_track_xy_list[trial_idx][0]
    eye_track_ys = eye_track_xy_list[trial_idx][1]

    trial_length = min(len(track_it_xs), len(eye_track_xs)/2)
    print 'Plotting trial ' + str(trial_idx) + ' with length ' + str(trial_length) + '.'

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
    trackit_line, = ax.plot([], [], lw=2)
    eye_track_line, = ax.plot([], [], lw=2)
    distractors_lines = []
    for j in range(len(distractors)):
        distractors_lines.extend(ax.plot([], [], 'r', lw=2))
    
    # initialization function: plot the background of each frame
    def init():
        plt.axes(xlim=(x_min, x_max),ylim=(y_min, y_max))
        trackit_line.set_data([], [])
        eye_track_line.set_data([], [])
        for j in range(len(distractors)):
            distractors_lines[j].set_data([], [])
        return trackit_line, eye_track_line,

    # i is the index of the current time point in track-it (30 Hz)
    # Recall that eyetracking moves twice as fast (60 Hz)
    #in range(trial_length - lag)
    def animate(i):
        trackit_line.set_data(track_it_xs[i:(i + lag)], track_it_ys[i:(i + lag)])
        eye_track_line.set_data(eye_track_xs[(2*i):(2*(i + lag))], eye_track_ys[(2*i):(2*(i + lag))])
        for j in range(len(distractors)):
            distractors_lines[j].set_data(distractors[j][0][i:(i + distractor_lag)],
                                          distractors[j][1][i:(i + distractor_lag)])
        plt.draw()
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        timestep = 0.0333
        time.sleep(timestep)
        # print 'Frame: ' + str(i) + ' Error: ' + str(error_fcn(eye_track_xs[2*i], eye_track_ys[2*i], track_it_xs[i], track_it_ys[i]))
        return trackit_line, eye_track_line,


    anim = animation.FuncAnimation(fig, animate,
                                   init_func = init,
                                   frames = trial_length - lag,
                                   interval = 20,
                                   blit = False,
                                   repeat = False)
    plt.show()
