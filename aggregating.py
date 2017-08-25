import csv
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
import math
import numpy as np
import actual_scoring_copy as asc

trendline_list_0dis = []
trendline_list_same = []
trendline_list_diff = []
maindir = "adult_pilot/"; subject_id_list = ["Anna"]#,"Clara","Eden","Hyunji","Kristen","Nick","Rebeka","Sara"]
# maindir = "3yo/"; subject_id_list = ["A232","A233","A239","A240","A241","A242","A243","A244","A245","A247"]
# maindir = "blinky_pilot/"; subject_id_list = ["A197","A250","A252","A256","A257"]
filter_threshold = 1 # between 0 and 1; higher value keeps more data

for subject_id in subject_id_list:
    new_trendline = asc.read_and_analyze_file(maindir,"AllSame", subject_id, filter_threshold = filter_threshold)
    if new_trendline != None and len(new_trendline) > 0:
        trendline_list_same.append(new_trendline)
    new_trendline = asc.read_and_analyze_file(maindir,"AllDiff", subject_id, filter_threshold = filter_threshold)
    if new_trendline != None and len(new_trendline) > 0:
        trendline_list_diff.append(new_trendline)
    new_trendline = asc.read_and_analyze_file(maindir,"0Distractors", subject_id, filter_threshold = filter_threshold)
    if new_trendline != None and len(new_trendline) > 0:
        trendline_list_0dis.append(new_trendline)

def plot_all_trials(trendline_list):
  for trendline in trendline_list:
    plt.plot(trendline)

def plot_all_trials_and_mean(trendline_list):
  print 'np.shape(trendline_list): ' + str(np.shape(trendline_list))
  for trendline in trendline_list:
    plt.plot(trendline, color = 'blue', alpha = 0.4)
  avg = map(np.nanmean,zip(*trendline_list))
  plt.plot(range(len(avg)), avg, color = 'red', linewidth = 2)

plt.subplot(2,3,1)
plot_all_trials(trendline_list_0dis)
plt.title('0 Dis')
# plt.xlabel('Frames')
plt.ylabel('Pixels')

plt.subplot(2,3,2)
plot_all_trials(trendline_list_same)
plt.title('All Same')
# plt.xlabel('Frames')
plt.ylabel('Pixels')

plt.subplot(2,3,3)
plot_all_trials(trendline_list_diff)
plt.title('All Diff')
# plt.xlabel('Frames')
plt.ylabel('Pixels')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
plt.legend(subject_id_list)

plt.subplot(2,3,4)
plot_all_trials_and_mean(trendline_list_0dis)
# plt.title('0 Dis Averaged')
plt.xlabel('Frames')
plt.ylabel('Pixels')

plt.subplot(2,3,5)
plot_all_trials_and_mean(trendline_list_same)
# plt.title('All Same Averaged')
plt.xlabel('Frames')
plt.ylabel('Pixels')

plt.subplot(2,3,6)
plot_all_trials_and_mean(trendline_list_diff)
# plt.title('All Diff Averaged')
plt.xlabel('Frames')
plt.ylabel('Pixels')

#plt.subplot(3,2,5)
#plt.plot(range(len(diff_avg)),diff_avg,color='red',linewidth=2,label='diff')
#plt.plot(range(len(same_avg)),same_avg,color='blue',linewidth=2,label='same')
#plt.legend(loc='upper right')
#plt.title('Same vs. Diff')
    
plt.show()
