import numpy as np

def get_trackit_MLE(eye_track, target, distractors, sigma2 = None):
  N = eye_track.shape[1] # length of trial, in frames
  num_distractors = distractors.shape[0]

  MLE = np.zeros(N, dtype = int)
  all_distances = np.zeros(num_distractors + 1)

  for n in range(N):
    if np.isnan(eye_track[0, n]):
      MLE[n] = -1
    else:
      all_distances[0] = np.linalg.norm(eye_track[:, n] - target[:, n])
      for k in range(num_distractors):
        all_distances[k + 1] = np.linalg.norm(eye_track[:, n] - distractors[k, :, n])
      MLE[n] = np.argmin(all_distances)
  return MLE
