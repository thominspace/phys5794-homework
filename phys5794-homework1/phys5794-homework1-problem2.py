# Thomas Edwards
# PHYS 5794 - Computational Physics
# 1/27/16
# Homework 1, Problem 2

# Problem statement:
# Write a program to calculate the first-order and second-order derivatives of f(x) = x^3 cos(x) at
# pi/2 <= x <= pi and estimate the numerical accuracy. You may use the three-point formulas for the
# calculations of the first-order and second-order derivatives. Use 100 uniform intervals in the range.
# For the boundary points, you may use the formulas discussed in the class or extrapolated values by
# using the linear interpolation or the Lagrange interpolation. The numerical accuracy at each x value
# can be obtained by comparing with the analytical result. (15 pts)

# usage: python phys5794-homework1-problem2.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib.pyplot import figure, axis, plot, title, xlabel, ylabel, savefig, legend


def main():

    # set up known information
    h = np.pi*.5/100
    x_range = np.arange(np.pi*.5, np.pi, np.pi*.5/100)
    fx = (x_range**3)*(np.cos(x_range))
    fx_len = len(x_range)

    # Extended fx using endpoints found using neville's method interpolation (from problem 1, slightly modified)
    # Note that we're only using 5 points in this interpolation.
    endpoints = np.array([x_range[0]-h, x_range[-1]+h])
    fx_endpoints = np.array([neville_interp(endpoints[0],
                                            x_range[0:5],
                                            fx[0:5]),
                             neville_interp(endpoints[1],
                                            x_range[len(x_range)-5:len(x_range)],
                                            fx[len(x_range)-5:len(x_range)])])
    fx = np.append(fx_endpoints[0], np.append(fx, fx_endpoints[1]))  # now we have an array with endpoints

    # first derivative
    fx_prime = (fx[2:fx_len+2] - fx[0:fx_len])/(2.*h)

    # second derivative
    fx_prime_prime = (fx[2:fx_len+2] - 2.*fx[1:fx_len+1] + fx[0:fx_len])/(h**2)

    # recover original fx array (since we put on some extra values)
    fx = fx[1:len(fx)-1]

    # actual derivatives of fx, found analytically
    fx_prime_actual = (x_range**2)*(3*np.cos(x_range)-x_range*np.sin(x_range))
    fx_prime_prime_actual = (x_range**2)*(-x_range*np.cos(x_range)-4*np.sin(x_range))+2*x_range*(3*np.cos(x_range)-x_range*np.sin(x_range))

    # errors found by comparing the analytic result to the numerical result
    max_error_fx_prime = max(abs(fx_prime_actual - fx_prime))
    max_percent_error_fx_prime = max(abs(fx_prime_actual - fx_prime)/abs(fx_prime_actual))*100.
    max_error_fx_prime_prime = max(abs(fx_prime_prime_actual - fx_prime_prime))
    max_percent_error_fx_prime_prime = max(abs(fx_prime_prime_actual - fx_prime_prime)/abs(fx_prime_actual))*100.
    print 'Max Absolute f\'(x) Error: ', max_error_fx_prime
    print 'Percent f\'(x) Error: ', max_percent_error_fx_prime, ' %'
    print 'Max Absolute f\'\'(x) Error: ', max_error_fx_prime_prime
    print 'Percent f\'\'(x) Error: ', max_percent_error_fx_prime_prime, ' %'

    # plot result
    # "x" markers used for the analytic result, to compare
    figure(1)
    xlabel('$x$')
    ylabel('$f(x)$, $f\'(x)$, $f\'\'(x)$')
    title('Numerical Differentiation Example')
    plot(x_range, fx, 'r', label='$f(x)$')
    plot(x_range, fx_prime, 'g', label='$f\'(x)$')
    plot(x_range, fx_prime_actual, 'gx')
    plot(x_range, fx_prime_prime, 'b', label='$f\'\'(x)$')
    plot(x_range, fx_prime_prime_actual, 'bx')
    legend(numpoints=1, loc='upper left')
    savefig('homework1_problem2_plot1.png')


def neville_interp(user_eval, user_x, user_fx):

    maximum_number_of_generations = len(np.atleast_1d(user_x))-1
    closest_neighbor_index = (np.abs(user_x-user_eval)).argmin()  # finding the closest x value index that is known
    # find the value via recursion
    return neville_recursion_workhorse(user_eval, user_x, user_fx, user_fx,
                                       maximum_number_of_generations, closest_neighbor_index)


def neville_recursion_workhorse(user_eval, x, last_c, last_d,
                                gen_max, neighbor_ix,
                                m=0, result=0, error_estimation_c=0, error_estimation_d=0):

    m += 1 # increase the generation count
    elements_this_gen = gen_max-m+1

    # allocate the next generation
    next_c = np.zeros(elements_this_gen)
    next_d = np.zeros(elements_this_gen)

    # calculate the next generation
    for ix in range(elements_this_gen):
        next_c[ix] = ((user_eval-x[ix])/(x[ix+m]-x[ix]))*(last_c[ix+1]-last_d[ix])
        next_d[ix] = ((user_eval-x[ix+m])/(x[ix]-x[ix+m]))*(last_d[ix]-last_c[ix+1])

    # calculations for error estimations (running total)
    error_estimation_c += last_c[0]
    error_estimation_d += last_d[0]

    # Start from the closest point and work down the tree. Once you reach the bottom, work your way up.
    if neighbor_ix < elements_this_gen+1:
        result += last_c[neighbor_ix]
    else:
        result += last_d[-1]

    # go to the next generation, or if you're in the last generation start the recursive return
    if m == gen_max+1:
        error_estimation = .5*(abs(error_estimation_c) + abs(error_estimation_d))
        return result
    else:
        return neville_recursion_workhorse(user_eval, x, next_c, next_d,
                                           gen_max, neighbor_ix,
                                           m, result, error_estimation_c, error_estimation_d)


if __name__ == "__main__":
    main()