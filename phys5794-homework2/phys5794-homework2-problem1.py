# Thomas Edwards
# PHYS 5794 - Computational Physics
# 2/8/16
# Homework 1, Problem 1

# Problem statement:
# Consider a quantum particle confined in a one-dimensional box with size L (for example, one nanometer)
# where a potential energy is zero inside the box but finite (positive value) outside the box. In this
# finite-square well problem, when the energy of the particle is positive but less than the magnitude of
# the finite potential energy, the total energy of the particle is quantized. This quantized energy can
# be graphically obtained. For even solutions, one needs to solve tan(x) = a/x for x graphically, where
# x is related to the energy of the particle. For a = 5, write a program to solve this equation within
# (pi/2, 3pi/2), by using the bisection method. In the interval, pi/2 and 3pi/2 are not included. There
# exists only one solution in (pi/2, 3pi/2). Try several different sets of initial intervals for bracketing and
# check if your final answer does not depend on them. After solving the equation, you must confirm
# that your numerical solution satisfies the equation within the numerical accuracy. (10 pts).

# usage: python phys5794-homework2-problem1.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def bisection(left_guess, right_guess, tol=1e-15, max_iterations=10000):

    # initialize the endpoints using the guesses
    a = left_guess
    b = right_guess
    iterations = 0

    # check if either guess is already within tolerance
    if abs(fx(left_guess)) < tol:
        return left_guess
    if abs(fx(right_guess)) < tol:
        return right_guess

    # find the midpoint (bisected 'x') and save it, and find the function value at that point
    root_x = (right_guess + left_guess)/2.
    root_val = fx(root_x)

    # keep bisecting until you find a value within tolerance.
    # also includes a safeguard in case you don't have a root
    while iterations < max_iterations:
        if abs(fx(a)) < tol:
            return a
        elif abs(fx(b)) < tol:
            return b
        # find the bisection
        root_x = (a + b)/2.
        root_val = fx(root_x)
        # change the bounds to reflect the bisected value
        if fx(a)*root_val < 0.:
            b = root_x
        elif fx(b)*root_val < 0.:
            a = root_x
        iterations += 1


def fx(x):

    # return the function value
    return np.tan(x)-(4./x)


def main():

    # do the solve
    left_guess = np.pi/2.+.01
    right_guess = 3.*np.pi/2.-.01
    root_x = bisection(np.pi/2.+.01, 3.*np.pi/2.-.01)
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
    plt.savefig('homework2_problem1_plot1.png')

    # check a bunch of different endpoints
    print 'Using Range: [', np.pi/2.+.01, ', ', 3.*np.pi/2.-.01, ']'
    print 'Root at x =', bisection(np.pi/2.+.1, 3.*np.pi/2.-.01)
    print 'Using Range: [', 3.0, ', ', 4.5, ']'
    print 'Root at x =', bisection(3.0, 4.5)
    print 'Using Range: [', 2.5, ', ', 4., ']'
    print 'Root at x =', bisection(2.5, 4.)


if __name__ == "__main__":
    main()