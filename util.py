import math
import numpy as np

# Euclidean distance error
def error_fcn(et_x, et_y, trackit_x, trackit_y):
    return math.sqrt((et_x - trackit_x)**2 + (et_y - trackit_y)**2)

def interpolate_to_length(X, new_len):
  old_len = X.shape[0]
  return np.interp([(float(n)*old_len)/new_len for n in range(new_len)], range(old_len), X)
