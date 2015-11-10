"""
USAGE: ipython superimposed_sin.py [number of freq per band]
"""
import ipdb

import numpy as np
from numpy.random import uniform
np.random.seed(seed=438)

import os
import sys

""" If it was slow, use:
import multiprocessing
try:
    NUM_CPUS = multiprocessing.cpu_count()*3/4
except NotImplementedError:
    # arbitrary default
    NUM_CPUS = 4
from joblib import delayed, Parallel
data = Parallel(n_jobs=NUM_CPUS)(delayed(np.sin)(x) for x in batch)
"""

train_size, valid_size, test_size = 400000, 80000, 20000
#train_size, valid_size, test_size = 40, 8, 2
total_size = train_size + valid_size + test_size

save_path = '/Tmp/mehris/'

# low frequencies (with higher energy)
low_freq_band = [0.5, 5.]
low_amp_band = [0.5, 1.5]
# higher frequencies (with much less energy)
high_freq_band = [5.5, 10.]
high_amp_band = [0., 0.2]

# number of freq in each band
num_freq = int(sys.argv[1])

period = np.linspace(0, 2*np.pi, 256)
period = period.astype('float32')

# LOW freq sins
all_freq = uniform(low=low_freq_band[0],
                   high=low_freq_band[1],
                   size=num_freq*total_size)
all_amp = uniform(low=low_amp_band[0],
                  high=low_amp_band[1],
                  size=num_freq*total_size)[:, None]
sin = np.array([np.sin(2.*np.pi*freq*period) for freq in all_freq])
sin = sin.astype('float32')
sin *= all_amp
sin = sin.reshape((num_freq, total_size, -1)).sum(axis=0)

train = sin[:train_size, :]
train_mean = train.mean()
train_std = train.std()
train -= train_mean
train /= train_std

valid = sin[train_size:train_size+valid_size, :]
valid -= train_mean
valid -= train_std

test = sin[train_size+valid_size:, :]
test -= train_mean
test /= train_std

file_name = 'sin_lowfreq_{0}_train-valid-test-mean-std'.format(num_freq)
save_at = os.path.join(save_path, file_name)
print "saving low..."
print save_at
np.savez(save_at,\
         train=train, valid=valid, test=test,\
         train_mean=train_mean, train_std=train_std)
print "done."

# HIGH freq sins
all_freq = uniform(low=high_freq_band[0],
                   high=high_freq_band[1],
                   size=num_freq*total_size)
all_amp = uniform(low=high_amp_band[0],
                  high=high_amp_band[1],
                  size=num_freq*total_size)[:, None]
sin = np.array([np.sin(2.*np.pi*freq*period) for freq in all_freq])
sin = sin.astype('float32')
sin *= all_amp
sin = sin.reshape((num_freq, total_size, -1)).sum(axis=0)

train = sin[:train_size, :]
train_mean = train.mean()
train_std = train.std()
train -= train_mean
train /= train_std

valid = sin[train_size:train_size+valid_size, :]
valid -= train_mean
valid -= train_std

test = sin[train_size+valid_size:, :]
test -= train_mean
test /= train_std

file_name = 'sin_highfreq_{0}_train-valid-test-mean-std'.format(num_freq)
save_at = os.path.join(save_path, file_name)
print "saving high..."
print save_at
np.savez(save_at,\
         train=train, valid=valid, test=test,\
         train_mean=train_mean, train_std=train_std)
print "done."

""" dumb plotter:
import matplotlib.pyplot as plt
low = np.load(os.path.join(save_path, 'sin_lowfreq_{0}_train-valid-test-mean-std'.format(num_freq)+'.npz'))
low_tr = low['train']
low_tr_mean = low['train_mean']
low_tr_std = low['train_std']
ind=0
plt.plot((sin[ind]*train_std+train_mean)+(low_tr[ind]*low_tr_std+low_tr_mean))
plt.plot(low_tr[ind]*low_tr_std+low_tr_mean+0.5)
plt.plot((sin[ind]*train_std+train_mean)+1.)
plt.show()
ipdb.set_trace()
ind=1
plt.plot((sin[ind]*train_std+train_mean)+(low_tr[ind]*low_tr_std+low_tr_mean))
plt.plot(low_tr[ind]*low_tr_std+low_tr_mean+0.5)
plt.plot((sin[ind]*train_std+train_mean)+1.)
plt.show()
"""
