# Thomas Edwards
# PHYS 5794 - Computational Physics
# 2/16/16
# Homework 3, Problem 1

# Problem statement:
# Write a program to solve the following linear algebraic equations using the LU decomposition method
# with implicit partial pivoting. Compare your numerical result with the answer, x = 224/1545,
# y = -268/1545, z = 83/515, and u = 589/1545. Don't forget carrying out the same row permutations
# (for pivoting) to the right-hand side vector b. This step is important or obtaining the solution. In
# the report, the following six issues should be addressed:
#
# (i): Write down the constructed lower L and upper triangular matrices U.
# (ii): Check if L . U = A. If you have to carry out row permutations or operations, one needs to check
# if L . U = P . A, where P is a permutation matrix. You may use the 3 x 3 matrix example used in the
# class or other simple examples for debugging.
# (iii): Write down a permutation matrix P if it was used.
# (iv): Write down the solution x = (x, y, z, u)
# (v): Confirm if your numerical solution satisfies the given answer.
# (vi): Now turn off the pivoting procedure and see if you obtain the same answer with the pivoting. If
# so, compare the two answers up to seven decimal places. If not, explain the reason.
#
# The equations of interest are:
#
# 2x + 3y + 10z - u = 1
# 10x + 15y + 3z + 7u = 2
# -4x + y + 2z + 9u = 3
# 15x - 3y + z + 3u = 4

# usage: python phys5794-homework3-problem1.py &

# Imports
import numpy as np


def lu(a, b, pivot=1):

    # copy the original A and b, just in case we need them
    original_a = np.copy(a)
    original_b = np.copy(b)

    # allocate the permutation matrix
    p = np.eye(a.shape[0])

    # allocate l and u
    l = np.eye(a.shape[0])
    u = np.zeros(a.shape)

    # we assume a square, n by n matrix, so find n now
    n = a.shape[0]

    # Pivot
    if pivot == 1:
        for columns in range(n):
            this_column = abs(a[columns:,columns])
            max_index = np.argmax(this_column)
            swap_rows(a, max_index+columns, columns)
            swap_rows(b, max_index+columns, columns)
            swap_rows(p, max_index+columns, columns)

    # full LU decomposition.
    for i in range(n):
        for j in range(i, n):
            sum_subtotal_u = 0.
            for k in range(i):
                sum_subtotal_u += l[i, k]*u[k, j]
            u[i, j] = a[i, j] - sum_subtotal_u

        for j in range(i+1, n):
            sum_subtotal_l = 0.
            for k in range(j):
                sum_subtotal_l += l[j, k]*u[k, i]
            l[j, i] = (a[j, i] - sum_subtotal_l)/u[i, i]

    # prove that LU = PA (visually)
    lu = np.dot(l, u)
    pa = np.dot(p, original_a)
    print '*****************************************'
    print '               L and U                   '
    print '*****************************************'
    print ' L = '
    print l
    print ' U = '
    print u
    print '*****************************************'
    print '  LU, PA, and Original A for Comparison  '
    print '*****************************************'
    print ' LU = '
    print lu
    print ' PA = '
    print pa
    print ' A (Original) = '
    print original_a
    if pivot == 1:
        print '*****************************************'
        print '                  P                      '
        print '*****************************************'
        print ' P = '
        print p

    # find solution: Ly = b, Ux = y
    y = np.zeros(n)
    y[0]= b[0]/l[0, 0]
    for i in range(1, n):
        sum_subtotal_y = 0.
        for k in range(i):
            sum_subtotal_y += l[i, k]*y[k]
        y[i] = (b[i] - sum_subtotal_y)/l[i, i]

    print '*****************************************'
    print '                   y                     '
    print '*****************************************'
    print ' y = '
    print y

    solution = np.zeros(n)
    solution[-1] = y[-1]/u[-1, -1]
    this_range = n-np.arange(n-1)-2
    for i in this_range:
        sum_subtotal_solution = 0.
        for k in range(i+1, n):
            sum_subtotal_solution += u[i, k]*solution[k]
        solution[i] = (y[i] - sum_subtotal_solution)/u[i, i]

    print '*****************************************'
    print '                    x                    '
    print '*****************************************'
    print ' x (solution) = '
    print solution


def swap_rows(matrix, index_1, index_2):

    # assumes NxN matrix
    temp = np.copy(matrix[index_2])
    matrix[index_2] = matrix[index_1]
    matrix[index_1] = temp
    return


def main():

    # simple test case from lecture notes

    test_a = np.array([[1., 1., 4.,],
                      [3., 2., 1.],
                      [2., 3., 1.]])
    test_b = np.array([12., 11., 13.])
    test_known_solution = np.array([1., 3., 2.])

    # set the initial data
    problem_a = np.array([[2., 3., 10., -1.],
                        [10., 15., 3., 7.],
                        [-4., 1., 2., 9.],
                        [15., -3., 1., 3.]])
    problem_b = np.array([1., 2., 3., 4.])
    known_solution = np.array([224./1545.,
                               -268./1545.,
                               83./515.,
                               589./1545.])

    # do the decomposition

    print '-----------------------------------------'
    print '    A simple test case, using pivoting    '
    print '-----------------------------------------'
    lu(test_a, test_b)
    print '*****************************************'
    print '       Comparison to Known Solution      '
    print '*****************************************'
    print test_known_solution
    print '\n\n\n'

    print '-----------------------------------------'
    print '    Same test case, without pivoting     '
    print '-----------------------------------------'
    lu(test_a, test_b, pivot=0)
    print '*****************************************'
    print '       Comparison to Known Solution      '
    print '*****************************************'
    print test_known_solution
    print '\n\n\n'

    print '-----------------------------------------'
    print '       Problem Statement Solution        '
    print '-----------------------------------------'
    lu(problem_a, problem_b)
    print '*****************************************'
    print '       Comparison to Known Solution      '
    print '*****************************************'
    print known_solution
    print '\n\n\n'

    print '-----------------------------------------'
    print '  Problem Statement Solution (no pivot)  '
    print '-----------------------------------------'
    lu(problem_a, problem_b, pivot=0)
    print '*****************************************'
    print '       Comparison to Known Solution      '
    print '*****************************************'
    print known_solution


if __name__ == "__main__":
    main()