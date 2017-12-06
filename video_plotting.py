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

root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/" # Laptop
# root = "/home/sss1/Desktop/academic/projects/eyetracking/" # Home desktop
subject_type = "adult_pilot/" # "3yo/" 
TI_data_dir = "TrackItOutput/AllDiff/"
ET_data_dir = "EyeTracker/AllDiff/"
TI_fname = "shashank.csv" # "AnnaSame.csv" # "A232Same.csv" 
ET_fname = "shashank1_12_5_2017_13_41.csv" # "AnnaSame_9_13_2016_13_25.csv" # "A232Same_3_29_2016_10_26.csv" 

TI_file_path = root + subject_type + TI_data_dir + TI_fname
ET_file_path = root + subject_type + ET_data_dir + ET_fname

print 'Track-It file: ' + TI_file_path

track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

# boundaries of track-it grid
x_min = 400
x_max = 1600
y_min = 000
y_max = 1000

space = 50 # number of extra pixels to display on either side of the plot

# First set up the figure, the axis, and the plot element we want to animate
lag = 10 # plot a time window of length lag, so we can see the trajectory
print 'Number of trials: ' + str(len(track_it_xy_list))

# gbbgbgg
# for trial_idx in range(len(track_it_xy_list)):
for trial_idx in [0,3,5,6]:#range(1):

  # x and y lists for the current trial
  # Recall that eyetracking (60 Hz) uses twice the frequency of TrackIt (30 Hz)
  # We adjust for this by interpolating TrackIt to have the same number of
  # frames as eyetracking
  N = len(eye_track_xy_list[trial_idx][0]) # Number of eye-tracking frames in trial
  N_short = 700 # TODO: REMOVE; TEMP DEBUGGING VARIABLE TO AVOID NANS
  eye_track = np.array(eye_track_xy_list[trial_idx])[:, 0:N_short]
  target = np.zeros((2, N_short))
  target[0,:] = interpolate_to_length(np.array(track_it_xy_list[trial_idx][0]), N)[0:N_short]
  target[1,:] = interpolate_to_length(np.array(track_it_xy_list[trial_idx][1]), N)[0:N_short]
  distractors_old = np.array(distractors_xy_list[trial_idx])
  distractors_old = distractors_old
  distractors = np.zeros(distractors_old.shape[0:2] + (N_short,))
  for k in range(distractors.shape[0]):
    distractors[k,0,:] = interpolate_to_length(np.array(distractors_old[k,0,:]), N)[0:N_short]
    distractors[k,1,:] = interpolate_to_length(np.array(distractors_old[k,1,:]), N)[0:N_short]

  MLE, X, mu = getMLE(eye_track, target, distractors)
  print MLE


  # if not np.allclose(X.transpose(), eye_track):
  #   raise ValueError
  if not np.allclose(mu[1:mu.shape[0], :, :].swapaxes(1,2), distractors):
    raise ValueError

  trial_length = target.shape[1]
  print 'Plotting trial ' + str(trial_idx) + ' with length ' + str(trial_length) + '.'

  fig = plt.figure()
  ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
  trackit_line, = ax.plot([], [], lw = 2)
  eye_track_line, = ax.plot([], [], lw = 2)
  distractors_lines = []
  for j in range(len(distractors)):
    distractors_lines.extend(ax.plot([], [], 'r', lw = 2))
  state_point = ax.scatter([], [], s = 50)
  
  # initialization function: plot the background of each frame
  def init():
    # plt.axes(xlim=(x_min, x_max),ylim=(y_min, y_max))
    # trackit_line.set_data([], [])
    # eye_track_line.set_data([], [])
    # for j in range(len(distractors)):
    #   distractors_lines[j].set_data([], [])
    return# trackit_line, eye_track_line,

  # show frames in range(trial_length - lag)
  def animate(i):
    trackit_line.set_data(target[0,i:(i + lag)], target[1,i:(i + lag)])
    eye_track_line.set_data(eye_track[0,i:(i + lag)], eye_track[1,i:(i + lag)])
    for j in range(len(distractors)):
      distractors_lines[j].set_data(distractors[j][0][i:(i + lag)],
                                    distractors[j][1][i:(i + lag)])
    state = MLE[i + lag]
    if state == 0:
      state_point.set_offsets(target[:,i + lag - 1])
    else:
      state_point.set_offsets(distractors[state - 1, :, i + lag - 1])
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
