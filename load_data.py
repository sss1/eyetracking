import csv
import numpy as np

# root = "/home/painkiller/Desktop/academic/projects/trackit/eyetracking/blinky_pilot/"
# TI_data_dir = "TrackItOutput/0Distractors/"
# ET_data_dir = "EyeTracker/0Distractors/"
# TI_fname = "A256_0Dis.csv"
# ET_fname = "A256_0Dis_2_1_2017_14_32.csv"

# # bondaries of track-it grid
# x_min = 0 #sys.maxint
# x_max = 2000 #-sys.maxint - 1
# y_min = 0 #sys.maxint
# y_max = 2000 #-sys.maxint - 1

# track_it_xy_list, trials_time_list = read_TI_data(TI_file_path):
def read_TI_data(TI_file_path):

  flag = 0
  start = 0
  end = 0
  index = 0 # target index
  absolute_start = 0
  startTime_flag = 0
  add_absolute = False
  trials_time_list = []
  trial_x_list = []
  trial_y_list = []
  track_it_xy_list = []
  distractors_x_list = []
  distractors_y_list = []
  distractors_xy_list = []

  with open(TI_file_path, 'rb') as TI_file:
    reader = csv.reader(TI_file, delimiter = ',')

    for row in reader:
      if startTime_flag == 1:
          absolute_start = float(row[2])
          startTime_flag = 0
      if len(row)>0 and row[0] == "END New Trial":
          trials_time_list.append([start, end])
          track_it_xy_list.append([trial_x_list,trial_y_list])
          distractors_xy_list.append(list(zip(map(list, zip(*distractors_x_list)),map(list, zip(*distractors_y_list)))))
          flag = 0
    
      if flag != 0:
          if flag == 1:
              if add_absolute:
                  start = float(row[0])+absolute_start
              else:
                  start = float(row[1])
              trial_x_list = []
              trial_y_list = []
              distractors_x_list = []
              distractors_y_list = []
              flag = 2
    
          trial_x_list.append(float(row[index]))        
          trial_y_list.append(float(row[index+1]))
    
          temp_x = []
          temp_y = []
          if add_absolute:
              end = float(row[0])+absolute_start
              for i in range(1,len(row)):
                  if i % 2 == 1 and i != index:
                      temp_x.append(float(row[i]))
                  elif i % 2 == 0 and i != index+1:
                      temp_y.append(float(row[i]))
          else:
              end = float(row[1])
              for i in range(2,len(row)):
                  if i % 2 == 0 and i != index:
                      temp_x.append(float(row[i]))
                  elif i % 2 == 1 and i != index+1:
                      temp_y.append(float(row[i]))
    
          # x_min = min(x_min, min(temp_x))
          # x_max = max(x_max, max(temp_x))
          # y_min = min(y_min, min(temp_y))
          # y_max = max(y_max, max(temp_y))
          distractors_x_list.append(temp_x)
          distractors_y_list.append(temp_y)
    
      if "target" in row:
          index = row.index("target")
      if len(row) > 0 and row[0] == "Frame Timestamp (Relative)":
          flag = 1
      if len(row) > 0 and row[0] == "Frame Timestamp":
          flag = 1
          add_absolute = True
      if "startTime" in row:
          startTime_flag = 1

  return track_it_xy_list, distractors_xy_list, trials_time_list

# Examples:
#   No Filtering:
#     eye_track_xy_list = read_ET_data(ET_file_path, trials_time_list)
#   Filter trials with <50% valid eye-tracking data
#     eye_track_xy_list = read_ET_data(ET_file_path, trials_time_list, filter_threshold = 0.5)
# Keeps trials with at least filter_threshold fraction of valid eye-tracking points
def read_ET_data(ET_file_path, trials_time_list, filter_threshold = 1.0):

  last_valid_x = 0
  last_valid_y = 0
  trial_count = 0

  count_invalid = 0

  et_x_list = []
  et_y_list = []
  eye_track_xy_list = []
  trials_to_keep = []

  with open(ET_file_path, 'rb') as ET_file:

    reader = csv.reader(ET_file, delimiter = ',')
    for row in reader:
  
      # If we're done reading all the trials, break
      if trial_count >= len(trials_time_list):
          break
      
      # rows containing valid eyetracking data should have len(row) >= 7
      if len(row) < 7:
          continue
  
      # Read in eyetrack data between trial start and trial end
      trial_start_time = trials_time_list[trial_count][0]
      trial_end_time = trials_time_list[trial_count][1]
      current_et_time = float(row[0])

      # Eyetracking data before trial starts is only used for interpolating
      x_mean, y_mean, x_left, x_right, y_left, y_right = (np.asarray(row[1:7])).astype(np.float)
      # print 'x_mean: ' + str(x_mean) + ' y_mean: ' + str(y_mean) + ' x_left: ' + str(x_left) + ' ...'
      # float(row[1])
      # float(row[2])
      # float(row[3])
      # float(row[5])
      # float(row[4])
      # float(row[6])
      if min(x_left, x_right, y_left, y_right) != 0:
        last_valid_x = x_mean
        last_valid_y = y_mean
  
      # Skip eyetracking data before trial starts
      if current_et_time < trial_start_time:
          continue
  
      # If trial hasn't ended yet, read in next row of eyetracking data
      if current_et_time < trial_end_time:
  
        # Fill in any missing points with the last previous valid point
        if min(x_left, x_right, y_left, y_right) != 0:
          last_valid_x = x_mean
          last_valid_y = y_mean
        else:
          x_mean = last_valid_x
          y_mean = last_valid_y
          count_invalid += 1
  
        et_x_list.append(x_mean)
        et_y_list.append(y_mean)
  
      # Done reading trial
      # If there's enough valid data, append to overall data
      # Either way, reset and prepare to read next trial
      else:
        if count_invalid <= len(et_x_list) * filter_threshold:
          print 'Keeping trial ' + str(trial_count) + ' with error rate ' + str(float(count_invalid)/len(et_x_list)) + '.'
          trials_to_keep.append(trial_count)
          eye_track_xy_list.append([et_x_list, et_y_list])
        else:
          print 'Discarding trial ' + str(trial_count) + ' with error rate ' + str(float(count_invalid)/len(et_x_list)) + '.'
        et_x_list = []
        et_y_list = []
        trial_count += 1
        count_invalid = 0

  return eye_track_xy_list, trials_to_keep

# Given trackit data and a list of trials to keep (e.g., based on filtering
# eye-tracking data), return a subset of data containing only those trials
def filter_trackit(track_it_xy_list, to_keep):
  track_it_xy_list = [track_it_xy_list[trial] for trial in to_keep]
  return track_it_xy_list

def load_full_subject_data(TI_file_path, ET_file_path, filter_threshold = 1.0):
  track_it_xy_list, distractors_xy_list, trials_time_list = read_TI_data(TI_file_path)
  eye_track_xy_list, trials_to_keep = read_ET_data(ET_file_path, trials_time_list, filter_threshold = filter_threshold)
  track_it_xy_list = filter_trackit(track_it_xy_list, trials_to_keep)
  distractors_xy_list = filter_trackit(distractors_xy_list, trials_to_keep)
  return track_it_xy_list, distractors_xy_list, eye_track_xy_list
