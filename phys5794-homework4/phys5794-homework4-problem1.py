# Thomas Edwards
# PHYS 5794 - Computational Physics
# 2/22/16
# Homework 4, Problem 1

# Problem statement:
# Solve the following linear algebraic equations by using the singular value decomposition (SVD)
# method. As you noticed, the equations are the same as those in HW3. This is a chance to solve
# the same problem with a different method. For diagonalization of a matrix, use the Jacobi method
# discussed in the class. At least, the following items must be discussed in the report: (i) the column-
# orthogonal matrix U , the square diagonal matrix W , and the orthogonal matrix V . (ii) Show that
# A = UWV T . (iii) Write down the solution from your code. (iv) Confirm that your solution satisfies
# the original set of the equations. Is the solution of this problem the same as that in HW3? (v)
# The number of iterations used for the Jacobi transformation, where one iteration means after you
# go over all the off-diagonal elements once. (vi) The tolerance used in the Jacobi transformation.
# (vii) Confirm that the Jacobi transformation diagonalizes the given matrix which you would like to
# diagonalize. For both the SVD method and the Jacobi method, you need to mention how you tested
# your code.
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


def svd(a, b, pivot=1):

    # copy the original A and b, just in case we need them
    original_a = np.copy(a)

    # get dimension of matrix (we assume N x N)
    n = a.shape[0]

    [v, eigen_v] = jacobi(np.dot(np.transpose(original_a), original_a))
    [v, eigen_v] = fix_zeros(v, eigen_v)
    print '*****************************************'
    print '           V and Eigenvalues             '
    print '*****************************************'
    print ' V = '
    print v
    print ' Eigenvalues = '
    print eigen_v
    print '*****************************************'
    print '                   W                     '
    print '*****************************************'
    w = np.zeros((n, n))
    over_w_sqrt = np.zeros((n, n))
    for ix in range(len(eigen_v)):
        if eigen_v[ix] > 0:
            w[ix, ix] = np.sqrt(eigen_v[ix])
            over_w_sqrt[ix, ix] = 1./np.sqrt(eigen_v[ix])
    print ' W = '
    print w
    print ' 1/sqrt(W) = '
    print over_w_sqrt
    print '*****************************************'
    print '                   U                     '
    print '*****************************************'
    u = gen_u(original_a, over_w_sqrt, v)
    [u, eigen_v] = fix_zeros(u, eigen_v)
    print ' U = '
    print u
    print '*****************************************'
    print '         Testing is A = U W V_T          '
    print '*****************************************'
    print ' A = '
    print original_a
    print ' U W V_T = '
    check_a = np.dot(u, np.dot(w, np.transpose(v)))
    print check_a
    print '*****************************************'
    print '                   x                     '
    print '*****************************************'
    x = np.dot(v, np.dot(over_w_sqrt, np.dot(np.transpose(u), b)))
    print x
    print '*****************************************'
    print '          Check if Ax = b                '
    print '*****************************************'
    print ' Ax = '
    print np.dot(original_a, x)
    print ' b = '
    print b


def gen_u(user_a, user_w, user_v):

    # generates U as w_i A v_i, as per lecture notes
    a = np.copy(user_a)
    w = np.copy(user_w)
    v = np.copy(user_v)

    n = user_v.shape[0]
    u = np.zeros((n, n))
    for ix in range(n):
        u[:, ix] = w[ix, ix]*np.dot(a, v[:, ix])

    return u


def fix_zeros(v, e, tol=1e-10):

    # This is a safegap to make sure that certain matrices and vectors are zeroed

    # copy original data
    e_copy = np.copy(e)
    v_copy = np.copy(v)

    # fix zeros
    for ix in range(v_copy.shape[0]):
        for jx in range(v_copy.shape[1]):
            if abs(v_copy[ix, jx]) < tol:
                v_copy[ix, jx] = 0.
    for ix in range(e_copy.shape[0]):
        if abs(e_copy[ix]) < tol:
            e_copy[ix] = 0.

    return v_copy, e_copy


def jacobi(a, max_sweeps=50, tol=1e-13):

    # array dimensions
    n = a.shape[0]

    # keep number of sweeps counted
    number_of_sweeps = 0

    # the permuatation matrix
    v = np.eye(n)

    # sweep over all indices in array
    for ix in range(max_sweeps):
        # check if we are done by using a boolean
        clean_sweep = True
        for ip in range(n-1):
            for iq in range(ip+1, n):
                # find the largest element
                if abs(a[ip, iq]) > tol:
                    clean_sweep = False
                    [a, v] = jacobi_workhorse(a, v, ip, iq)
                # zero out values below tolerance
                if abs(a[ip, iq]) < tol:
                    a[ip, iq] = 0.
        number_of_sweeps += 1
        if clean_sweep:
            break
        if ix == max_sweeps-1:
            print 'Maximum sweeps have been reached.'

    # eignevalues are the diagonal of transformed A
    eigen = np.diagonal(a)

    # print things for problem
    print '*****************************************'
    print '     Number of Sweeps (Iterations)       '
    print '*****************************************'
    print number_of_sweeps
    print '*****************************************'
    print '             Diagonalized A              '
    print '*****************************************'
    print a

    return v, eigen


def jacobi_workhorse(a, v, p, q, tol=1e-13):

    # copy a, since we need it to not be overwritten during permutation
    a_prime = np.copy(a)
    v_prime = np.copy(v)

    # find the dimensions (we assume an NxN matrix)
    n = a_prime.shape[0]

    # check tolerance again
    if abs(a[p, q]) > tol:
        if abs(a[p, q]) < abs(a[q, q] - a[p, p])*tol:
            t = a[p, q]/(a[q, q] - a[p, p])  # correction based on estimation, in case a[p, q] is very small
        else:
            theta = (a[q, q] - a[p, p])/(2.*a[p, q])
            t = 1./(abs(theta) + np.sqrt(theta**2 + 1.))
            if theta < 0.: t = -t  # for some reason sign(theta) didn't work but his did!

        # find sine, cosine, and tau
        c = 1./np.sqrt(1. + (t**2))
        s = t*c
        tau = s/(1. + c)

        # make the rotation adjustments
        a_prime[p, q] = 0.
        a_prime[q, p] = 0.
        a_prime[p, p] = a[p, p] - t*a[p, q]
        a_prime[q, q] = a[q, q] + t*a[p, q]

        for r in range(n):  # elimination modifications
            if r < p:
                a_prime[r, p] = a[r, p] - s*(a[r, q] + tau*a[r, p])
                a_prime[r, q] = a[r, q] + s*(a[r, p] - tau*a[r, q])
            if r > q:
                a_prime[p, r] = a[p, r] - s*(a[q, r] + tau*a[p, r])
                a_prime[q, r] = a[q, r] + s*(a[p, r] - tau*a[q, r])
            if r > p and r < q:
                a_prime[p, r] = a[p, r] - s*(a[r, q] + tau*a[p, r])
                a_prime[r, q] = a[r, q] + s*(a[p, r] - tau*a[r, q])

        for i in range(n):  # rotation matrix modifications
            v_prime[i, p] = v[i, p] - s*(v[i, q] + tau*v[i, p])
            v_prime[i, q] = v[i, q] + s*(v[i, p] - tau*v[i, q])

    return a_prime, v_prime

def main():

    np.seterr(all='raise')

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

    # test to make sure it works
    print '-----------------------------------------'
    print ' A simple test case, from lecture notes  '
    print '-----------------------------------------'
    print '*****************************************'
    print '           Starting A and b              '
    print '*****************************************'
    print ' A = '
    print test_a
    print ' b = '
    print test_b
    svd(test_a, test_b)

    print '*****************************************'
    print '             Known Solution:             '
    print '*****************************************'
    print test_known_solution

    print '\n\n\n'

    # do the decomposition
    print '-----------------------------------------'
    print '       Problem Statement Solution        '
    print '-----------------------------------------'
    print '*****************************************'
    print '           Starting A and b              '
    print '*****************************************'
    print ' A = '
    print problem_a
    print ' b = '
    print problem_b
    svd(problem_a, problem_b)

    print '*****************************************'
    print '             Known Solution:             '
    print '*****************************************'
    print known_solution


if __name__ == "__main__":
    main()