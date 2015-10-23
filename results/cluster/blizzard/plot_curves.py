
from blocks.serialization import load
import sys
import os
import glob
import ipdb

from pandas import DataFrame
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

experiment_num = 48324
repetition_num = 10 #range(1,11)

channels = [['valid_nll'], ['train_nll']]
num_plots = len(channels)

fig, axarr = pyplot.subplots(num_plots, sharex=True)
save_dir = os.environ['RESULTS_DIR']

for iteration in range(repetition_num):
    job_id = str(experiment_num) + str(iteration+1)
    exp_path = os.path.join(save_dir,'blizzard/', job_id)

    for file_ in os.listdir(exp_path):
        if file_.endswith(".pkl"):
            exp_file = file_

    main_loop = load(os.path.join(exp_path,exp_file))
    log = main_loop.log

    del main_loop

    #ipdb.set_trace()
    df = DataFrame.from_dict(log, orient='index')

    if num_plots > 1:
        for i, channel in enumerate(channels):
        	#axarr[0] = pyplot.plot(df[channel], label = iteration)
            df[channel].plot( ax = axarr[i], label = iteration)
    else:
        df[self.channels[0]].plot()

axarr[0].legend(range(repetition_num), loc='best')
axarr[1].legend(range(repetition_num), loc='best')

#ipdb.set_trace()

axarr[1].set_ylim(axarr[0].get_ylim())

axarr[0].set_title("Validation set")
axarr[1].set_title("Training set")

pyplot.savefig(os.path.join(save_dir,'blizzard/', str(experiment_num) + "_summary.png"))
pyplot.close()






