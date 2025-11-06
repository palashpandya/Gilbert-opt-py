# In this script we will investigate the effect of mixing the 3 qubit W-state
# with the maximally mixed state, rho[p] = p W + (1-p) I/8, for different values
# of p. We will use parallel processing to process all the points on the convex
# combination, to figure out at what value of p, the state becomes fully separable.
# Old value from the literature is 0.177.

import time
from os import path
import numpy as np

from scipy import io

from functions import gilbert_only_dist
import configparser
from math import prod
from matplotlib import pyplot as plt
from multiprocessing import Pool




# def par_func(p):
#     css, dist, trials, dist_list = gilbert(p * rho + (1 - p) * max_mix, dim_list, 25, 1000000, opt_state="on")
#     return dist

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    inpFile = config['DEFAULT']['inputState']
    outFile = config['DEFAULT']['outputState']
    outFile2 = config['DEFAULT']['outputStatistics']
    dim_list = np.fromstring(config['DEFAULT']['dimList'], dtype=int, sep=' ')
    max_iter = int(config['DEFAULT']['maxIterations'])
    max_trials = int(config['DEFAULT']['maxTrials'])
    #dim_list2 = np.array([2, 2, 2])
    rho = np.array([])
    rho1 = np.array([])
    max_mix = np.identity(prod(dim_list),complex)
    if path.exists(inpFile):
        rho = io.mmread(inpFile)
    else:
        print("ERROR: Unable to find the input matrix. Check the name supplied in the config.")
        exit(0)
    #pdist = []



    start = time.time()
    inp = [float(x+1)/10. for x in range(10)]
    ppd = []
    with Pool() as prc:
        ppd_res = [prc.apply_async(gilbert_only_dist, args=(p * rho + (1 - p) * max_mix, dim_list, 100, 10000000, "on")) for p in inp]
        ppd = [ar.get() for ar in ppd_res]
    print(ppd)
    end  = time.time()
    print("time: ",end-start)
    plt.plot([x**0.5 for x in ppd])
    plt.plot(ppd)
    plt.show()


#[0.006511700672920808, 0.02590155380346068, 0.05782300958768831, 0.10161613701499893, 0.1559205764384808, 0.21761347607481818, 0.2795070170761502, 0.33016714003485864, 0.36099774037774285, 0.42976868223627596]