import csv
import math
from data_paths_full import shrinky_ages_file as ages_path
import numpy as np

def read_ages(has_performance = False):
  shrinky_ages = []
  noshrinky_ages = []
  if has_performance:
    shrinky_performance = []
    noshrinky_performance = []
  with open(ages_path, 'rb') as csvfile:
    age_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    passed_header = False
    for row in age_reader:
      if not passed_header: # skip header row
        passed_header = True
        continue

      # row[0] is the subject ID

      new_val = float(row[1])
      if not math.isnan(new_val): # skip nans
        shrinky_ages.append(new_val)

      new_val = float(row[2])
      if not math.isnan(new_val): # skip nans
        noshrinky_ages.append(new_val)

      if has_performance: # if file also contains mean performance for each subject/condition, read that too
        new_val = float(row[3])
        if not math.isnan(new_val): # skip nans
          shrinky_performance.append(new_val)

        new_val = float(row[4])
        if not math.isnan(new_val): # skip nans
          noshrinky_performance.append(new_val)

  if has_performance:
    return np.asarray(shrinky_ages), np.asarray(noshrinky_ages), np.asarray(shrinky_performance), np.asarray(noshrinky_performance)
  return np.asarray(shrinky_ages), np.asarray(noshrinky_ages)
