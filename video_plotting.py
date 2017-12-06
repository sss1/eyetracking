import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from load_data import load_full_subject_data
from eyetracking_hmm import getTrackItMLE
from util import interpolate_to_length_D, impute_missing_data_D

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

track_it_xy_list, distractors_xy_list, eye_track_xy_list \
  = load_full_subject_data(TI_file_path,
                            ET_file_path,
                            filter_threshold = 1)

# boundaries of track-it grid
x_min = 400
x_max = 1600
y_min = 0
y_max = 1000

space = 50 # number of extra pixels to display on either side of the plot

# First set up the figure, the axis, and the plot element we want to animate
lag = 10 # plot a time window of length lag, so we can see the trajectory # added 1 and 2
trials_to_show = range(len(track_it_xy_list))
print 'Number of trials: ' + str(len(trials_to_show))

for trial_idx in trials_to_show:

  # x and y lists for the current trial
  # Recall that eyetracking (60 Hz) uses twice the frequency of TrackIt (30 Hz)
  # We adjust for this by interpolating TrackIt to have the same number of
  # frames as eyetracking
  N = len(eye_track_xy_list[trial_idx][0]) # Number of eye-tracking frames in trial
  eye_track = impute_missing_data_D(np.array(eye_track_xy_list[trial_idx]))[:, 0:N]
  target = interpolate_to_length_D(np.array(track_it_xy_list[trial_idx]), N)[:,0:N]
  distractors_old = np.array(distractors_xy_list[trial_idx])
  distractors_old = distractors_old
  distractors = np.zeros(distractors_old.shape[0:2] + (N,))
  for k in range(distractors.shape[0]):
    distractors[k,:,:] = interpolate_to_length_D(np.array(distractors_old[k,:,:]), N)[:,0:N]

  MLE, X, mu = getTrackItMLE(eye_track, target, distractors)
  print MLE

  trial_length = target.shape[1]
  print 'Plotting trial ' + str(trial_idx) + ' with length ' + str(trial_length) + '.'
 
  # initializate plot background and objects to plot
  fig = plt.figure()
  ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
  trackit_line, = ax.plot([], [], lw = 2)
  eye_track_line, = ax.plot([], [], lw = 2)
  distractors_lines = []
  for j in range(len(distractors)):
    distractors_lines.extend(ax.plot([], [], 'r', lw = 2))
  state_point = ax.scatter([], [], s = 50)

  # Rather than a single point, show tail of object trajectories (frames in range(trial_length - lag))
  # This makes it much easier to follow objects visually
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
    # time.sleep(timestep)
    return trackit_line, eye_track_line,


  anim = animation.FuncAnimation(fig, animate,
                                 frames = trial_length - lag,
                                 interval = 20,
                                 blit = False,
                                 repeat = False)
  plt.show()
