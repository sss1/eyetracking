import math

# Euclidean distance error
def error_fcn(et_x, et_y, trackit_x, trackit_y):
    return math.sqrt((et_x - trackit_x)**2 + (et_y - trackit_y)**2)

