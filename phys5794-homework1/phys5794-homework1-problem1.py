# Thomas Edwards
# PHYS 5794 - Computational Physics
# 1/27/16
# Homework 1, Problem 1

# Problem statement:
# The following five data points are given: f(0.2) = 0.0015681, f(0.3) = 0.0077382, f(0.4) = 0.023579,
# f(0.5) = 0.054849, f(0.6) = 0.10696. This problem is about writing a code to evaluate f(0.16) and
# f(0.46) by using the Lagrange interpolation, i.e., a fourth-order polynomial interpolation function
# (25 pts).
# Write a program for the Lagrange interpolation by using the Neville's method. Compute f(0.16)
# and f(0.46) and estimate their numerical uncertainties or numerical errors. (20 pts)
# Compare the above result to the result using a linear interpolation. (5 pts)

# usage: python phys5794-homework1-problem1.py &

# Imports
import numpy as np
import heapq
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib.pyplot import figure, axis, plot, title, xlabel, ylabel, savefig, legend


def lagrange_interp(user_eval, user_x, user_fx):

    # recast input as numpy arrays
    user_eval_np = np.atleast_1d(user_eval)
    user_x_np = np.atleast_1d(user_x)
    user_fx_np = np.atleast_1d(user_fx)

    # allocate result
    neville_interpolated_result = np.zeros(len(user_eval_np))
    neville_interpolated_error = np.zeros(len(user_eval_np))
    linear_interpolated_result = np.zeros(len(user_eval_np))
    linear_interpolated_error = np.zeros(len(user_eval_np))

    # set up plot for the data
    figure(1)
    plot(user_x, user_fx, 'o', label='Given Data')
    axis([.1, .7, -.05, .15])
    xlabel('$x$')
    ylabel('$f(x)$')
    title('Given Data and Interpolated Values')

    # handle mismatched x and f(x)
    number_of_data_points_x = len(user_x_np)
    number_of_data_points_fx = len(user_fx_np)
    if number_of_data_points_x != number_of_data_points_fx:
        raise ValueError('x and f(x) are not the same length.')

    # do Neville's Algorithm for each point given by user
    result_index = 0
    print 'Neville Interpolated Values:'
    for eval_index in user_eval_np:
        [neville_interpolated_result[result_index], neville_interpolated_error[result_index]] = \
            neville_interp(eval_index, user_x_np, user_fx_np)
        print "x: ", eval_index, \
            " f(x): ", neville_interpolated_result[result_index], \
            ' error: ', neville_interpolated_error[result_index]
        result_index += 1
    plot(user_eval_np, neville_interpolated_result, 'or', label='Interpolated Points')

    # compare to linear interpolation
    print 'Comparison to Linear:'
    result_index = 0
    for eval_index in user_eval_np:
        [linear_interpolated_result[result_index], linear_interpolated_error[result_index]] = \
            linear_interp(eval_index, user_x_np, user_fx_np)
        print 'x: ', eval_index, \
            ' f(x): ', linear_interpolated_result[result_index], \
            ' error: ', linear_interpolated_error[result_index]
        result_index += 1
    plot(user_eval_np, linear_interpolated_result, 'xk', label='(Linearly) Interpolated Points')

    # save plot
    legend(numpoints=1, loc='upper left')
    savefig('homework1_problem1_plot1.png')


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
        return [result, error_estimation]
    else:
        return neville_recursion_workhorse(user_eval, x, next_c, next_d,
                                           gen_max, neighbor_ix,
                                           m, result, error_estimation_c, error_estimation_d)


def linear_interp(user_eval, user_x, user_fx):

    # find the two closest points
    [closest_x, closest_fx] = find_two_closest_data_points(user_eval, user_x, user_fx)

    # *Technically*, the Neville version for 2 points *should* be linear. If you want to not
    # look at a specifically-coded-linear-interpolation, comment the "linear_interp_workhorse"
    # function out and uncomment the "neville_interp" function.

    # return neville_interp(user_eval, closest_x, closest_fx)  # <-- This one
    return linear_interp_workhorse(user_eval, closest_x, closest_fx)


def linear_interp_workhorse(eval_point, x, fx):

    # linear interpolation should just be a one-liner, based on the formulas in the notes
    return [fx[0] + (fx[1] - fx[0])*((eval_point - x[0])/(x[1] - x[0])),  # interpolated value
            .5*(abs(fx[0] - eval_point) + abs(fx[1] - eval_point))]  # error estimation


def find_two_closest_data_points(user_eval, user_x, user_fx):

    # find the indices of the nearest two points
    diff_user_x = abs(user_x - user_eval)
    # this is just a fancy way of finding the nearest two values.
    nearest_two = heapq.nsmallest(2, range(len(diff_user_x)), key=diff_user_x.__getitem__)
    return [user_x[nearest_two], user_fx[nearest_two]]


def main():

    # set of the problem using the known parameters
    x_data_known = [0.2,0.3,0.4,0.5,0.6]
    fx_data_known = [0.0015681,0.0077382,0.023579,0.054849,0.10696]
    points_to_solve_for = [0.16, 0.46]

    # do the solve
    lagrange_interp(points_to_solve_for, x_data_known, fx_data_known)


if __name__ == "__main__":
    main()