import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import config
import workbook
import annotator

sys.path.append("..")

try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')

plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.home'].remove('r')

names  = np.load("samples.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inputs for continuing work and providing a name.")
    parser.add_argument("--continue",   dest="continue_previous", action="store_true", help="Continue previous work")
    parser.add_argument("--downsample", dest="downsample", action="store_true", help="downsample the images to half size")
    parser.add_argument("name", type=str, default = None, nargs="?", help="The name to associate with the work")
    args = parser.parse_args()

    if args.downsample:
        config.DOWNSAMPLE = True

    # Call the function with the parsed arguments.
    c = workbook.open_workbook(args.continue_previous, args.name)
else:
    c = workbook.open_workbook()


## start looping through samples: 
while c < config.MAX_SAMPLES:
    workbook.update_sample(c)
    
    if config.CACHE_MASKS:
        ant = annotator.SampleAnnotator(c, labels, names)
        print(c)
        c += 1
        continue

    ant = annotator.SampleAnnotator(c, labels, names)
    ant.run()
 

    workbook.save_metrics(c, ant.score, ant.ng, ant.nr, ant.stdx, ant.stdy, ant.gp, ant.rp, ant.msk, ant.attempt)
    workbook.update_survey()

    contin = input("do u want to continue? press y if you want to continue or anyting otherwise ")
    if not contin == 'y':
        workbook.save_workbook()
        break

    c += 1
    print("Sample:", c)
