import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from load_data import load_full_subject_data
from eyetracking_hmm import getTrackItMLE

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

# boundaries of track-it grid
x_min = 400
x_max = 1600
y_min = 0
y_max = 1000

space = 50 # number of extra pixels to display on either side of the plot

# First set up the figure, the axis, and the plot element we want to animate
lag = 10 # plot a time window of length lag, so we can see the trajectory # added 1 and 2
trials_to_show = range(len(target_all_trials))
print 'Number of trials: ' + str(len(trials_to_show))

for trial_idx in trials_to_show:

  eyetrack = eyetrack_all_trials[trial_idx]
  target = target_all_trials[trial_idx]
  distractors = distractors_all_trials[trial_idx]

  MLE = getTrackItMLE(eyetrack, target, distractors)
  print MLE

  trial_length = target.shape[1]
  print 'Plotting trial ' + str(trial_idx) + ' with length ' + str(trial_length) + '.'
 
  # initializate plot background and objects to plot
  fig = plt.figure()
  ax = plt.axes(xlim=(x_min, x_max), ylim = (y_min, y_max))
  trackit_line, = ax.plot([], [], lw = 2)
  eyetrack_line, = ax.plot([], [], lw = 2)
  distractors_lines = []
  for j in range(len(distractors)):
    distractors_lines.extend(ax.plot([], [], 'r', lw = 2))
  state_point = ax.scatter([], [], s = 50)

  # Rather than a single point, show tail of object trajectories (frames in range(trial_length - lag))
  # This makes it much easier to follow objects visually
  def animate(i):
    trackit_line.set_data(target[0,i:(i + lag)], target[1,i:(i + lag)])
    eyetrack_line.set_data(eyetrack[0,i:(i + lag)], eyetrack[1,i:(i + lag)])
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
    return trackit_line, eyetrack_line,

  anim = animation.FuncAnimation(fig, animate,
                                 frames = trial_length - lag,
                                 interval = 20,
                                 blit = False,
                                 repeat = False)
  plt.show()
