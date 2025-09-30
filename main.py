# Minimum Hilbert-Schmidt Distance using Gilbert's algorithm
import time
from os import path
import numpy as np
# from numpy import ndarray, dtype, generic, bool_
from scipy import io
from functions import gilbert, generate_report
import configparser

if __name__ == '__main__':
    # rho = random_pure_dl([2, 2])
    # rho = make_density(rho)
    config = configparser.ConfigParser()
    config.read('config.ini')
    inpFile = config['DEFAULT']['inputState']
    outFile = config['DEFAULT']['outputState']
    outFile2 = config['DEFAULT']['outputStatistics']
    dim_list = np.fromstring(config['DEFAULT']['dimList'], dtype=int, sep=' ')
    max_iter = int(config['DEFAULT']['maxIterations'])
    max_trials = int(config['DEFAULT']['maxTrials'])
    print(inpFile,outFile,outFile2)
    dim_list2 = np.array([2, 2, 2])
    if path.exists(inpFile):
        rho = io.mmread(inpFile)
    else:
        print("ERROR: Unable to find the input matrix. Check the name supplied in the config.")
        exit(0)

    start = time.time()
    css, dist, trials, dist_list = gilbert(rho,  dim_list, max_iter, max_trials, opt_state="on")  # (, rho1_in=approx)
    stop = time.time()

    print(css, dist, trials, stop-start)
    io.mmwrite(outFile, css)
    np.savetxt(outFile2, dist_list)
    generate_report(dist_list)


