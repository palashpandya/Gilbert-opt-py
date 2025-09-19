#Export a numpy array as Matrix Market file (.mtx)
from scipy import io
from numpy import array

rho = array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(complex)

io.mmwrite("rho_in.mtx", rho)