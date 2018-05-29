import csv
import math
from data_paths_full import shrinky_ages_file as ages_path

def read_ages():
  shrinky_ages = []
  noshrinky_ages = []
  with open(ages_path, 'rb') as csvfile:
    age_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row_idx = 0
    for row in age_reader:
      if row_idx > 0: # skip header row
        new_val = float(row[1])
        if not math.isnan(new_val): # skip nans
          shrinky_ages.append(new_val)
        new_val = float(row[2])
        if not math.isnan(new_val): # skip nans
          noshrinky_ages.append(new_val)
      row_idx += 1
  return shrinky_ages, noshrinky_ages
