# Thomas Edwards
# PHYS 5794 - Computational Physics
# 1/27/16
# Homework 1, Problem 3

# Problem statement:
# Write a program to solve the following linear algebraic equations using the Gauss Elimination method
# with implicit partial pivoting. The implicit partial pivoting means:
# (i) First, all of the elements in each row are normalized by the largest coefficient in the row.
# (ii) Second, rows are interchanged with each other such that the diagonal elements become the largest
# (in magnitude) for each column. You are supposed to do pivoting column by column. Compare your
# numerical result with the answer, x = 58/85, y = -167/170, z = -78/85, and u = -5/2. Now turn off the
# pivoting procedure and check if your numerical answer without the pivoting differs from that with the
# pivoting. (20 pts)

# usage: python phys5794-homework2-problem3.py &

# Imports
import numpy as np


def main():

    print 'Gauss, with pivoting:'
    gauss()
    print 'Gauss, without pivoting:'
    gauss(pivot=0)


def gauss(pivot=1):

    # initialize
    a = np.array([[1., -1., -2., 1.], [3., 2., -1., 2.], [2., 3., 1., -1.], [10., -4., 3., 2.]])
    b = np.array([1., -4., 0., 3.])
    max_m = a.shape[0]

    # normalize
    for rows in range(a.shape[0]):
        largest_value_in_row = max(a[rows])
        largest_value_in_row = max(largest_value_in_row, b[rows])
        smallest_value_in_row = min(a[rows])
        smallest_value_in_row = min(smallest_value_in_row, b[rows])
        if abs(largest_value_in_row) > abs(smallest_value_in_row):
            a[rows] /= largest_value_in_row
            b[rows] /= largest_value_in_row
        else:
            a[rows] /= smallest_value_in_row
            b[rows] /= smallest_value_in_row

    # Pivot
    if pivot == 1:
        for columns in range(a.shape[1]):
            this_column = abs(a[columns:,columns])
            max_index = np.argmax(this_column)
            swap_rows(a, max_index+columns, columns)
            swap_rows(b, max_index+columns, columns)

    # Eliminate
    for columns in range(a.shape[1]):
        for rows_under in np.arange(columns+1, max_m):
            m = -a[rows_under, columns]/a[columns, columns]
            a[rows_under, columns] = 0.

            for columns_under in np.arange(columns+1, max_m):
                a[rows_under, columns_under] += a[columns, columns_under]*m

            b[rows_under] += b[columns]*m

    # solve
    x = np.zeros(max_m)
    k = max_m-1
    x[k] = b[k]/a[k,k]
    while k >= 0:
        x[k] = (b[k] - np.dot(a[k,k+1:],x[k+1:]))/a[k,k]
        k -= 1

    print 'A:'
    print a
    print 'b:'
    print b
    print 'x:'
    print x

    known_results = np.array([58./85., -167./170., -78./85., -5./2.])
    print 'Known Results: '
    print known_results

    return


def swap_rows(matrix, index_1, index_2):

    # assumes NxN matrix
    temp = np.copy(matrix[index_2])
    matrix[index_2] = matrix[index_1]
    matrix[index_1] = temp
    return


if __name__ == "__main__":
    main()