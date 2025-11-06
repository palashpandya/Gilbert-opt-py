import numpy as np
#from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def fit_func(x,a,b,c):
    return (b/x)**(c)+a


if __name__ == '__main__':
    dist = np.loadtxt('output_series.csv',delimiter=',',dtype=float)
    # Discard the first third
    red_dist = np.array([x for x in dist[int(2*len(dist)/4):]])
    #corrections
    X = np.array([ x+int(len(dist)/3) for x in range(len(red_dist))])
    popt, pcov = curve_fit(fit_func, X,red_dist)
    print(popt,pcov)
    pred = popt[0]
    plt.plot(X, [fit_func(x, *popt) for x in X], 'g-',
              label='fit: a=%f' % pred)
    plt.scatter(X,red_dist,color='red')
    plt.show()

