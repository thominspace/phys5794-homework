# Thomas Edwards
# PHYS 5794 - Computational Physics
# 4/2/16
# Homework 8, Problem 1

# Problem statement:

# usage: python phys5794-homework8-problem1.py &

# Imports
import numpy as np
import matplotlib
from random import random as rand
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def all_mc():

    n0 = 20.
    n1 = 100.
    b = 1.
    a = 0.
    total_points = (np.arange(15)+1)*10000
    integral = total_points*0.
    stds = total_points*0.
    for ix in np.arange(len(total_points)):

        sum, sum_squared = mc(total_points[ix], n1, n0)
        f = (1./((total_points[ix]-n1)/n0))*sum
        f_2 = (1./((total_points[ix]-n1)/n0))*sum_squared

        integral[ix] = sum*((b-a)/((total_points[ix]-n1)/n0))
        stds[ix] = (b-a)*np.sqrt((f_2-f**2.)/((total_points[ix]-n1)/n0))

        print '---------------'
        print 'Number of Points:'
        print total_points[ix]
        print 'Numerical Integral:'
        print integral[ix], '+/-', stds[ix]

    plt.plot((total_points-n1)/n0, stds, 'o', label='Standard Deviations')
    plt.plot((total_points-n1)/n0, .3/np.sqrt((total_points-n1)/n0), 'r', label='Scaled $1/\sqrt{M}$')
    plt.title('Standard Deviation and Scaled Total Points')
    plt.xlabel('Total Points')
    plt.ylabel('Standard Deviation')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem1_plot0.png')


def mc(number_of_samples, n1, n0):

    configs = np.zeros(number_of_samples)
    delta = .3
    configs[0] = rand()

    sum = 0.
    sum_squared = 0.
    for ix in np.arange(len(configs))-1:
        delta_x = delta*(2.*rand()-1)
        next_x = (configs[ix] + delta_x) % 1.
        if check_configuration(configs[ix], next_x, rand()):
            configs[ix+1] = next_x
        else:
            configs[ix+1] = configs[ix]
            print 'REJECTED'

        if ix >= n1:
            if not (ix-n1) % n0:
                sum += fx(configs[ix])
                sum_squared += fx(configs[ix])**2.

    return sum, sum_squared


def check_configuration(x1, x2, zeta):

    if zeta < (dist_fn(x2)/dist_fn(x1)):
        return True
    elif zeta >= (dist_fn(x2)/dist_fn(x1)):
        return False


def fx(x):
    return x**2.


def dist_fn(x):

    # uniform distribution function
    a = 0.
    b = 1.
    return 1./(b-a)


def main():

    all_mc()


if __name__ == "__main__":
    main()