# Thomas Edwards
# PHYS 5794 - Computational Physics
# 2/28/16
# Homework 5, Problem 1

# Problem statement:

# usage: python phys5794-homework5-problem1.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def lusol(a, b):

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

        # find solution: Ly = b, Ux = y
        y = np.zeros(n)
        y[0]= b[0]/l[0, 0]
        for i in range(1, n):
            sum_subtotal_y = 0.
            for k in range(i):
                sum_subtotal_y += l[i, k]*y[k]
            y[i] = (b[i] - sum_subtotal_y)/l[i, i]
        # print y

        solution = np.zeros(n)
        solution[-1] = y[-1]/u[-1, -1]
        this_range = n-np.arange(n-1)-2
        for i in this_range:
            sum_subtotal_solution = 0.
            for k in range(i+1, n):
                sum_subtotal_solution += u[i, k]*solution[k]
            solution[i] = (y[i] - sum_subtotal_solution)/u[i, i]

        return solution

    def swap_rows(matrix, index_1, index_2):

        # assumes NxN matrix
        temp = np.copy(matrix[index_2])
        matrix[index_2] = matrix[index_1]
        matrix[index_1] = temp
        return

    return lu(a, b)


def construct_lin_eqn_set(n, e_n, sigma_n):

    A_mat = np.array([[n], [sigma_n]])
    x_vec = np.copy(e_n)

    return A_mat, x_vec


def construct_chi_squared(a, b, x, y, sigma):

    chi_subtotal = 0.
    for ix in range(len(x)):
        chi_subtotal += ((y[ix] - a - b*x[ix])/(sigma[ix]))**2
    return chi_subtotal


def find_sums(x, y, sigma):

    s = 0.
    sx = 0.
    sy = 0.
    sxx = 0.
    sxy = 0.

    for index in range(len(x)):
        s += (1./(sigma[index]**2))
        sx += (x[index]/(sigma[index]**2))
        sy += (y[index]/(sigma[index]**2))
        sxx += ((x[index])**2/(sigma[index]**2))
        sxy += ((x[index]*y[index])/(sigma[index]**2))

    return [s, sx, sy, sxx, sxy]


def chi_squared_fit(x, y, sigma):

    # construct chi^2 and sums
    [s, sx, sy, sxx, sxy] = find_sums(x, y, sigma)

    # solve [alpha][a] = [beta] for a
    alpha = np.array([[s, sx], [sx, sxx]])
    beta = np.array([sy, sxy])
    solution = lusol(alpha, beta)
    print '*****************************************'
    print '        Values Obtained in Fitting       '
    print '*****************************************'
    print ' e_0 = '
    print solution[0]
    print ' Delta e = '
    print solution[1]

    # estimate errors
    delta = s*sxx-sx**2
    sigma_a = np.sqrt(sxx/delta)
    sigma_b = np.sqrt(s/delta)
    covariance = -sx/delta
    correlation = -sx/np.sqrt(sxx*s)

    print '*****************************************'
    print '          Standard Deviations           '
    print '*****************************************'
    print ' sigma_a = '
    print sigma_a
    print ' sigma_b = '
    print sigma_b
    print ' Covariance = '
    print covariance
    print ' Correlation = '
    print correlation

    # Q value calculation
    chi_squared = construct_chi_squared(solution[0], solution[1], x, y, sigma)
    print '*****************************************'
    print '                 Chi^2                   '
    print '*****************************************'
    print ' chi^2 = '
    print chi_squared

    nu = len(x) - 2
    q = 1. - small_gamma(nu/2., (chi_squared)/2.)/gamma(nu/2)
    print '*****************************************'
    print '                   Q                     '
    print '*****************************************'
    print ' Q = '
    print q

    return solution


def gamma(a):

    if a > 1.:
        return (a-1.)*gamma(a-1.)
    else:
        return a


def small_gamma(x, a, tol=1e-16, max_iter=100000):

    subtotal = np.exp(-x)*(x**a)*gamma(a)

    for n in np.arange(max_iter):
        addition = (x**n)/(gamma(a+1+n))
        if abs(addition) > tol:
            subtotal += addition
        else:
            return subtotal

    return subtotal


def main():

    # initialize the known data
    known_n = np.array([5., 6., 7., 8., 9., 10., 11.,
                        12., 13, 14., 15., 16., 17., 18.])
    known_e_n = np.array([8.206, 9.880, 11.50, 13.14, 14.82, 16.40, 18.04,
                         19.68, 21.32, 22.96, 24.60, 26.24, 27.88, 29.52])
    known_sigma_n = np.array([0.012, 0.015, 0.058, 0.025, 0.035, 0.013, 0.010,
                              0.039, 0.010, 0.019, 0.020, 0.011, 0.030, 0.048])

    # do the fitting
    print '-----------------------------------------'
    print '             Chi^2 Fitting               '
    print '-----------------------------------------'
    sol = chi_squared_fit(known_n, known_e_n, known_sigma_n)

    # plot solution to check validity
    x_range = np.arange(0., 20., .01)
    fx = sol[0] + sol[1]*x_range
    plt.figure(1)
    plt.plot(known_n, known_e_n, 'or', label='Data')  # f(x)
    plt.plot(x_range, fx, label='Fitting')  # estimated root
    plt.xlabel('$n$')
    plt.ylabel('$e_n$')
    plt.title('Data and $\chi^2$ fitting')
    plt.legend(numpoints=1, loc='upper left')
    plt.savefig('homework5_problem1_plot1.png')

    # Show difference between modelling and fitting
    print '*****************************************'
    print '   Differences between Data and Model    '
    print '*****************************************'
    print known_e_n-(sol[0] + sol[1]*known_n)

    # do the fitting again, but with sigma=1
    print '\n\n\n'
    print '-----------------------------------------'
    print '       Chi^2 Fitting (Sigma = 1)         '
    print '-----------------------------------------'
    sol_2 = chi_squared_fit(known_n, known_e_n, known_sigma_n*0+1.)

if __name__ == "__main__":
    main()