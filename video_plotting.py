import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
from load_data import load_full_subject_data

root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/adult_pilot/"
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "AnnaSame.csv"
ET_fname = "AnnaSame_9_13_2016_13_25.csv"
TI_file_path = root + TI_data_dir + TI_fname
ET_file_path = root + ET_data_dir + ET_fname

track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 0.000001)

# boundaries of track-it grid
x_min = 600
x_max = 1400
y_min = 100
y_max = 900

space = 50 # number of extra pixels to display on either side of the plot

# First set up the figure, the axis, and the plot element we want to animate
lag = 10 # plot a time window of length lag, so we can see the trajectory
distractor_lag = 5
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
    # ax = plt.axes(xlim=(min(eye_track_xs) - space, max(eye_track_xs) + space), ylim=(min(eye_track_ys) - space, max(eye_track_ys) + space))
    ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
    trackit_line, = ax.plot([], [], lw=2)
    eye_track_line, = ax.plot([], [], lw=2)
    distractors_lines = []
    for j in range(len(distractors)):
        distractors_lines.extend(ax.plot([], [], 'r', lw=2))
#     scat = ax.scatter(track_it_xs[lag - 1], track_it_ys[lag - 1], animated=True)

    
    # initialization function: plot the background of each frame
    def init():
        plt.axes(xlim=(x_min, x_max),ylim=(y_min, y_max))
        # plt.axes(xlim=(min(eye_track_xs) - space, max(eye_track_xs) + space), ylim=(min(eye_track_ys) - space, max(eye_track_ys) + space))
        # plt.axes(xlim=(min(x_min) - space, max(x_max) + space), ylim=(min(y_min) - space, max(y_max) + space))
        trackit_line.set_data([], [])
        eye_track_line.set_data([], [])
        for j in range(len(distractors)):
            distractors_lines[j].set_data([], [])
#         scat = ax.scatter(track_it_xs[lag - 1], track_it_ys[lag - 1], animated=True)
        return trackit_line, eye_track_line,

    # i is the index of the current time point in track-it (30 Hz)
    # Recall that eyetracking moves twice as fast (60 Hz)
#in range(trial_length - lag)
    def animate(i):
        # plt.cla()
        trackit_line.set_data(track_it_xs[i:(i + lag)], track_it_ys[i:(i + lag)])
        # plt.plot(track_it_xs[i:(i + lag)], track_it_ys[i:(i + lag)], color = "blue")
        # scat.set_offsets([track_it_xs[i + lag - 1], track_it_ys[i + lag - 1]])
        eye_track_line.set_data(eye_track_xs[(2*i):(2*(i + lag))], eye_track_ys[(2*i):(2*(i + lag))])
        # plt.plot(track_it_xs[i:(i + lag)], track_it_ys[i:(i + lag)], color = "blue")
        # plt.scatter(eye_track_xs[2*(i + lag) - 1], eye_track_ys[2*(i + lag) - 1], color = "green")
        for j in range(len(distractors)):
            distractors_lines[j].set_data(distractors[j][0][i:(i + distractor_lag)],
                                          distractors[j][1][i:(i + distractor_lag)])
            # plt.plot(distractors[j][0][i:(i + lag)], distractors[j][1][i:(i + lag)], color = "red")
        # plt.plot([x_min, x_max, x_max, x_min, x_min], [y_max, y_max, y_min, y_min, y_max], 'k')
        plt.draw()
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        timestep = 0.0333
        time.sleep(timestep)
        return trackit_line, eye_track_line,

    anim = animation.FuncAnimation(fig, animate, init_func = init,
                                   frames = trial_length - lag,
                                   interval = 20, blit = False,
                                   repeat=False)
    # print "trial to show: " + str(trial_idx)
    plt.show()
    # print "trial done: " + str(trial_idx)
    
# analysis.flush()
# analysis.close()
