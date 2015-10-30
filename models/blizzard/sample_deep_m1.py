from blocks.serialization import load
import os

save_dir = os.environ['RESULTS_DIR']
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "deep_m1_2"

main_loop=load(save_dir + experiment_name + ".pkl")

sampling_extension.do(0)
