# Thomas Edwards
# PHYS 5794 - Computational Physics
# 4/2/16
# Homework 8, Problem 2

# Problem statement:

# usage: python phys5794-homework8-problem2.py &

# Imports
import numpy as np
import matplotlib
from random import random as rand
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def all_mc():

    n0 = 25.
    n1 = 25.
    b = 1.
    a = 0.
    total_points = (np.arange(15)+1)*10000
    integral = total_points*0.
    stds = total_points*0.

    multiplier = (b-a)
    for ix in np.arange(len(total_points)):

        sum, sum_squared, these_total_points = mc(total_points[ix], n1, n0)
        f = (1./((total_points[ix]-n1)/n0))*sum
        f_2 = (1./((total_points[ix]-n1)/n0))*sum_squared

        integral[ix] = multiplier*(sum/these_total_points)
        stds[ix] = multiplier*np.sqrt((f_2-f**2.)/these_total_points)
                                                                                                                                                                                                                                                                                                                                                                                                                                    
        print '---------------'
        print 'Number of Points:'
        print total_points[ix]
        print 'Numerical Integral:'
        print integral[ix], '+/-', stds[ix]

    plt.figure(2)
    plt.plot((total_points-n1)/n0, stds, 'bo', label='Standard Deviations')
    plt.plot((total_points-n1)/n0, .3/np.sqrt((total_points-n1)/n0), 'r', label='Scaled $1/\sqrt{M}$')
    plt.title('Standard Deviation and Scaled Total Points')
    plt.xlabel('Total Points')
    plt.ylabel('Standard Deviation')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem2_plot1.png')


def mc(number_of_samples, n1, n0):

    configs = np.zeros(number_of_samples)
    delta = .3
    configs[0] = rand()

    total_points = 0.

    sum = 0.
    sum_squared = 0.
    for ix in np.arange(len(configs))-1:
        delta_x = delta*(2.*rand()-1)
        next_x = (configs[ix] + delta_x) % 1.
        if check_configuration(configs[ix], next_x, rand()):
            configs[ix+1] = next_x
        else:
            configs[ix+1] = configs[ix]

        if ix >= n1:
            if not (ix-n1) % n0:
                sum += fx(configs[ix])/dist_fn(configs[ix])
                sum_squared += (fx(configs[ix])/dist_fn(configs[ix]))**2.
                total_points += 1.

    return sum, sum_squared, total_points


def check_configuration(x1, x2, zeta):

    if zeta <= (dist_fn(x2)/dist_fn(x1)):
        return True
    elif zeta > (dist_fn(x2)/dist_fn(x1)):
        return False


def fx(x):
    return x**2.


def dist_fn(x):

    # distribution function
    c = 3./31.
    return c*((x**2.)+10.)


def auto_corr(l):
    m = 10000
    c1 = 0.
    c2 = 0.
    cn = 0.
    delta = .3

    samples = np.zeros(m)
    samples[0] = rand()
    for ix in np.arange(m)-1:
        delta_x = delta*(2.*rand()-1)
        next_x = (samples[ix] + delta_x) % 1.
        if check_configuration(samples[ix], next_x, rand()):
            samples[ix+1] = next_x
        else:
            samples[ix+1] = samples[ix]

    for ix in np.arange(m-l):
        c1 += (samples[ix]*samples[ix+l])
    c1 *= (1./(m-l))
    for ix in np.arange(m):
        cn += samples[ix]
        c2 += samples[ix]**2
    cn = ((1./m)*cn)**2
    c2 *= (1./m)

    return (c1-cn)/(c2-cn)


def main():

    norm_const = 3./31.
    print 'Normalization Constant:'
    print norm_const

    autocorr_tries = 200
    ls = np.zeros(autocorr_tries)
    for ix in np.arange(autocorr_tries):
        ls[ix] = auto_corr(ix)

    plt.figure(1)
    plt.plot(ls)
    plt.title('Autocorrelation Function')
    plt.xlabel('$l$')
    plt.ylabel('$C(l)$')
    plt.savefig('homework7_problem2_plot0.png')

    all_mc()


if __name__ == "__main__":
    main()