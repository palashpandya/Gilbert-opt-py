# Minimum Hilbert-Schmidt Distance using Gilbert's algorithm
import time
from os import path
import numpy as np
# from numpy import ndarray, dtype, generic, bool_
from scipy import io
from functions import gilbert
import configparser

# from prompt_toolkit.utils import to_str

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rho = random_pure_dl([2, 2])
    # rho = make_density(rho)
    config = configparser.ConfigParser()
    config.read('config.ini')
    inpFile = config['DEFAULT']['inputState'];
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

    rho1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(complex)
    print((dim_list))
    if rho.all()==rho1.all():
        print("TRUE")
    if dim_list2.all()==dim_list.all():
        print("TRUE")

    # rho = np.array([[0.5, 0, 0, 0, 0, 0, 0, 0.5],
    #                 [0, 0., 0 / 3., 0, 0 / 3., 0, 0, 0],
    #                 [0, 0 / 3., 0 / 3., 0, 0 / 3., 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0 / 3., 0 / 3., 0, 0 / 3., 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0.5, 0, 0, 0, 0, 0, 0, 0.5]]).astype(complex)
    # file_name = input('Enter the name for a matrix file (*.mtx): ')
    # if os.path.exists(file_name):
    #     rho = sp.io.mmread(file_name)
    # else:
    #     print("Unable to find the matrix.")
    # print(rho)

    start = time.time()
    css, dist, trials, dist_list = gilbert(rho,  dim_list, max_iter, max_trials, opt_state="on")  # (, rho1_in=approx)
    stop = time.time()

    print(css, dist, trials, stop-start)
    io.mmwrite(outFile, css)
    np.savetxt(outFile2, dist_list)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
