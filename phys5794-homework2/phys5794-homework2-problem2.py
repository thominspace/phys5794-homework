# Thomas Edwards
# PHYS 5794 - Computational Physics
# 1/27/16
# Homework 1, Problem 2

# Problem statement:
# Write a program to solve tan(x) = a/x for x in (pi/2, 3pi/2), where a = 5, by using the secant method.
# If necessary, you may combine the secant method with the bisection method. Compare your answer
# with that of Problem 1. (10 pts).

# usage: python phys5794-homework2-problem2.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def secant(left_guess, right_guess, tol=1e-13, max_iterations=10000):

    # initialize the endpoints using the guesses
    x_n = right_guess
    x_n_minus = left_guess
    iterations = 0

    # check if either guess is already within tolerance
    if abs(fx(left_guess)) < tol:
        return left_guess
    if abs(fx(right_guess)) < tol:
        return right_guess

    # check to see if the points are ok/startle a zero
    if fx(left_guess)*fx(right_guess) > 0.:
        print 'Bad guesses. Try again.'
        return

    # find the midpoint (bisected 'x') and save it, and use that as a starter
    root_x = (x_n + x_n_minus)/2.
    if fx(x_n)*fx(root_x) < 0.:
        x_n_minus = root_x
    elif fx(x_n_minus)*fx(root_x) < 0.:
        x_n = root_x

    # keep bisecting until you find a value within tolerance.
    # also includes a safeguard in case you don't have a root
    while iterations < max_iterations:
        if abs(fx(root_x)) < tol:
            return root_x
        # find the next root x value
        root_x = x_n - ((x_n - x_n_minus)/(fx(x_n) - fx(x_n_minus)))*fx(x_n)
        # determine the next bound, based on where the new bounds cross zero
        if fx(x_n)*fx(root_x) < 0.:
            x_n_minus = root_x
        elif fx(x_n_minus)*fx(root_x) < 0.:
            x_n = root_x
        iterations += 1


def fx(x):

    # return the function value
    return np.tan(x)-(4./x)


def main():

    # do the solve
    left_guess = np.pi/2.+.01
    right_guess = 3.*np.pi/2.-.01
    root_x = secant(np.pi/2.+.01, 3.*np.pi/2.-.01)
    print 'Found solution: ', root_x
    root_val = fx(root_x)

    # plot the function and the found root
    x_range = np.arange(left_guess, right_guess, .01)
    plt.figure(1)
    plt.plot(x_range, fx(x_range), label='$f(x)= \\tan(x)-\\frac{4}{x}$')  # f(x)
    plt.plot(root_x, root_val, 'or', label='Estimated Root')  # estimated root
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Function and Estimated Root or Closest Point')
    plt.legend(numpoints=1, loc='upper left')
    plt.savefig('homework2_problem2_plot1.png')

    # check a bunch of different endpoints
    print 'Using Range: [', np.pi/2.+.01, ', ', 3.*np.pi/2.-.01, ']'
    print 'Root at x =', secant(np.pi/2.+.1, 3.*np.pi/2.-.01)
    print 'Using Range: [', 3.0, ', ', 4.5, ']'
    print 'Root at x =', secant(3.0, 4.5)
    print 'Using Range: [', 2.5, ', ', 4., ']'
    print 'Root at x =', secant(2.5, 4.)


if __name__ == "__main__":
    main()