#Export a numpy array as Matrix Market file (.mtx)
from scipy import io
from numpy import array

# 3 qubit W STATE
rho = array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(complex)

# 3 qubit GHZ state
rho = array([[0.5, 0, 0, 0, 0, 0, 0, 0.5],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0.5, 0, 0, 0, 0, 0, 0, 0.5]]).astype(complex)

io.mmwrite("rho_inGHZ.mtx", rho)