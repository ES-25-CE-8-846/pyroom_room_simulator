import scipy.io as sio
from pathlib import Path
import numpy as np

mat = sio.loadmat(Path.home() / "Downloads" / "Test1_single_struct.mat")
mainStc: np.ndarray = mat['mainStc']
metaData = mat['metaData']

# These have shape (4,6)
phone: np.ndarray = mainStc[0,0][0]
bz: np.ndarray = mainStc[0,0][1]
intf: np.ndarray = mainStc[0,0][2]
dz: np.ndarray = mainStc[0,0][3]

bz_rir = np.transpose(bz[0,0], (0, 2, 1))
dz_rir = np.transpose(dz[0,0], (0, 2, 1))

print(bz_rir.shape, dz_rir.shape)


# VAST Hyperparams
J = 1028 # Filter length
mu = 1.0 # Importance on DZ power minimization
reg_param = 1e-5 # Regularization parameter
fs = 48000 # Sampling frequency

import test_filter_gen as tfg
filters = tfg.VAST(bz_rir, dz_rir, fs=fs, J=J, mu=mu, reg_param=reg_param)


# Calculate the acoustic contrast
from evaluation import acc_evaluation
from copy import deepcopy
print(f"filter shape:", filters["q_acc"].shape)
bz_rir_eval = bz_rir[np.newaxis, :, :, :]
dz_rir_eval = dz_rir[np.newaxis, :, :, :]
for name in ["q_acc", "q_vast", "q_pm"]:
    if filters[name] is not None:
        filter_eval = filters[name][np.newaxis, :, :]
        ac = acc_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval)) # Call the acc_evaluation function to calculate the acoustic contrast
        print(f"AC for {name} filter:", ac)
    
# For dirac delta filter
filter_eval = np.zeros_like(filter_eval)
filter_eval[0, :, int(np.ceil(J/2))] = 1.0
ac = acc_evaluation(filter_eval, deepcopy(bz_rir_eval), deepcopy(dz_rir_eval))
print(f"AC for dirac delta filter:", ac)

# Plot the filters
for name in ["q_acc", "q_vast", "q_pm"]:
    if filters[name] is not None:
        print(f"{name} filter shape:", filters[name].shape)
        tfg.plot_filters(filters[name], name)