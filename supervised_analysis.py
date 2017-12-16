from load_data import load_full_subject_data
import data_paths as dp
from eyetracking_hmm import get_trackit_MLE
from util import preprocess_all

data_child_super = [load_full_subject_data(*entry, filter_threshold = 1) for entry in zip(dp.trackit_fnames_child_supervised, dp.eyetrack_fnames_child_supervised)]

# Split off true labels from unsupervised data
true_labels = [subject_data[3] for subject_data in data_child_super]
data_child_super = [subject_data[0:3] for subject_data in data_child_super]

# Range of variance values to try
sigma2s = np.logspace(1, 4, num = 10)

# TODO: Iterate over sigma2s

data_child_super = [preprocess_all(*subject_data) for subject_data in data_child_super]
MLEs_child_super = [[get_trackit_MLE(*trial_data) for trial_data in zip(*subject_data)] for subject_data in data_child_super]
