import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
from load_data import load_full_subject_data
from util import error_fcn
from eyetracking_hmm import getMLE
from util import interpolate_to_length

# root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
subject_type = "adult_pilot/" # "3yo/" 
TI_data_dir = "TrackItOutput/AllSame/"
ET_data_dir = "EyeTracker/AllSame/"
TI_fname = "AnnaSame.csv" # "A232Same.csv" 
ET_fname = "AnnaSame_9_13_2016_13_25.csv" # "A232Same_3_29_2016_10_26.csv" 
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

# for trial_idx in range(len(track_it_xy_list)):
for trial_idx in range(1):

  MLE = getMLE()

  # x and y lists for the current trial
  # Recall that eyetracking (60 Hz) uses twice the frequency of TrackIt (30 Hz)
  # We adjust for this by interpolating TrackIt to have the same number of
  # frames as eyetracking
  eye_track_xs = np.array(eye_track_xy_list[trial_idx][0])
  eye_track_ys = np.array(eye_track_xy_list[trial_idx][1])
  N = len(eye_track_xs)
  N_short = 100
  eye_track_xs = eye_track_xs[0:N_short]
  eye_track_ys = eye_track_ys[0:N_short]
  track_it_xs = interpolate_to_length(np.array(track_it_xy_list[trial_idx][0]), N)[0:N_short]
  track_it_ys = interpolate_to_length(np.array(track_it_xy_list[trial_idx][1]), N)[0:N_short]
  distractors_old = np.array(distractors_xy_list[trial_idx])
  distractors = np.zeros((distractors_old.shape[0], distractors_old.shape[1], N_short))
  for k in range(len(distractors)):
    distractors[k,0,:] = interpolate_to_length(np.array(distractors_old[k,0,:]), N)[0:N_short]
    distractors[k,1,:] = interpolate_to_length(np.array(distractors_old[k,1,:]), N)[0:N_short]

  trial_length = len(track_it_xs)
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

  # show frames in range(trial_length - lag)
  def animate(i):
    trackit_line.set_data(track_it_xs[i:(i + lag)], track_it_ys[i:(i + lag)])
    eye_track_line.set_data(eye_track_xs[i:(i + lag)], eye_track_ys[i:(i + lag)])
    for j in range(len(distractors)):
      distractors_lines[j].set_data(distractors[j][0][i:(i + distractor_lag)],
                                    distractors[j][1][i:(i + distractor_lag)])
    plt.draw()
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    timestep = 0.0333 / 2
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
