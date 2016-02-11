# Thomas Edwards
# PHYS 5794 - Computational Physics
# 1/27/16
# Homework 1, Problem 3

# Problem statement:
# Write a program to calculate the integral
# exp(-x) dx, x = [0, 1]
# and estimate its numerical accuracy by using the Simpson rule. The numerical accuracy can be
# obtained by comparing with the analytical result. (10 pts)

# usage: python phys5794-homework1-problem3.py &

# Imports
import numpy as np


def main():

    # discretize the space with an EVEN number of points
    h = .01
    x_space = np.arange(0, 1, h) # 100 points
    fx = np.exp(-x_space)
    n = len(fx)

    # calculate integral using Simpson's rule
    S = 0.
    for l in range((n/2) - 1):
        S += fx[2*l] + 4.*fx[2*l+1] + fx[2*l+2]
    S *= (h/3.)

    # print the result
    print 'Integral Sum: ', S
    print 'Actual Integral: 0.63212 (Approximate)'
    print 'Difference: ', abs(S-0.63212)
    print 'Percent Difference: ', (abs(S-0.63212)/0.63212)*100, '%'


if __name__ == "__main__":
    main()