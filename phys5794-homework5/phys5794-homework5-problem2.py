# Thomas Edwards
# PHYS 5794 - Computational Physics
# 2/28/16
# Homework 5, Problem 2

# Problem statement:

# usage: python phys5794-homework5-problem2.py &

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


def construct_chi_squared(x, y, sigma, a, b):

    chi_subtotal = 0.
    for ix in range(len(x)):
        chi_subtotal += ((y[ix] - (a*x[ix]*np.exp(-b*x[ix])))/(sigma[ix]))**2
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


def small_gamma(x, a, tol=1e-16, max_iter=100):

    subtotal = np.exp(-x)*(x**a)*gamma(a)

    for n in np.arange(max_iter):
        addition = (x**n)/(gamma(a+1+n))
        if abs(addition) > tol:
            subtotal += addition
        else:
            return subtotal

    return subtotal


def construct_alpha_beta(x, y, sigma, a, k):

    # construct alpha and beta
    dchi2_da = 0.
    dchi2_dk = 0.
    d2chi2_da2 = 0.
    d2chi2_dadk = 0.
    d2chi2_dk2 = 0.

    for ix in range(len(x)):
        dy_da = x[ix]*np.exp(-k*x[ix])
        dy_dk = -a*(x[ix]**2)*np.exp(-k*x[ix])
        dy2_da2 = 0.
        dy2_dadk = -(x[ix]**2)*np.exp(-k*x[ix])
        dy2_dk2 = a*(x[ix]**3)*np.exp(-k*x[ix])
        fx = a*x[ix]*np.exp(-k*x[ix])

        dchi2_da += ((y[ix] - fx)/(sigma[ix]**2)) * dy_da
        dchi2_dk += ((y[ix] - fx)/(sigma[ix]**2)) * dy_dk
        d2chi2_da2 += (1./(sigma[ix]**2)) * dy_da * dy_da  #(dy_da**2 - (y[ix] - fx)*dy2_da2)
        d2chi2_dadk += (1./(sigma[ix]**2)) * dy_da * dy_dk  #(dy_da*dy_dk - (y[ix] - fx)*dy2_dadk)
        d2chi2_dk2 += (1./(sigma[ix]**2)) * dy_dk * dy_dk  #(dy_dk**2 - (y[ix] - fx)*dy2_dk2)

    alpha = np.array([[d2chi2_da2, d2chi2_dadk], [d2chi2_dadk, d2chi2_dk2]])
    beta = np.array([dchi2_da, dchi2_dk])
    return alpha, beta


def nonlinear_fitting(x, y, sigma, a, k, tol=1e-3, fudge_factor=0.0001, max_iter=100):

    # set up change in fudge factor
    fudge_reset = fudge_factor

    # find original chi^2([a])
    best_chi_squared_a = construct_chi_squared(x, y, sigma, a, k)

    [alpha, beta] = construct_alpha_beta(x, y, sigma, a, k)
    alpha_prime = np.copy(alpha)

    converged_flag = False
    good_sol_iter = -10
    iter_count = 0
    last_chi_squared = 0.

    while not converged_flag:

        # adjust alpha_prime
        for ix in range(alpha_prime.shape[0]):
            alpha_prime[ix, ix] *= (1. + fudge_factor)

        # solve for delta a
        delta_a_test = np.linalg.solve(alpha_prime, beta)
        delta_a = lusol(alpha_prime, beta)

        # now evaluate chi^2(a+delta_a)
        chi_squared_a_delta = construct_chi_squared(x, y, sigma, a+delta_a[0], k+delta_a[1])

        # If the new chi^2 is larger than best, increase the fudge factor
        if chi_squared_a_delta >= best_chi_squared_a:
            fudge_factor *= 10.  # delta_fudge
        # otherwise, save the new configuration
        if chi_squared_a_delta < best_chi_squared_a:
            # good value, so save the new fit parameters
            a += delta_a[0]
            k += delta_a[1]

            # lower the fudge factor
            fudge_factor = fudge_reset  # *= .1

            # save the new best chi^2
            best_chi_squared_a = construct_chi_squared(x, y, sigma, a, k)

            # construct new alpha and beta using new good fit values
            [alpha, beta] = construct_alpha_beta(x, y, sigma, a, k)
            alpha_prime = np.copy(alpha)

         # test if the change between steps is small enough
        if abs(last_chi_squared - chi_squared_a_delta) < tol:
            # if the last test was also good, we're done
            if good_sol_iter+1 == iter_count:
                converged_flag = True
            # if this is the first good test in a row, flag it and go again
            else:
                good_sol_iter = iter_count

        if iter_count > max_iter:
            print 'Max iterations reached. Breaking'
            break

        iter_count += 1
        last_chi_squared = chi_squared_a_delta

    # find the covariance and standard deviations using the inverse of the last alpha
    alpha_inv = np.linalg.inv(alpha)
    print '*****************************************'
    print '   Standard Deviations and Covariance    '
    print '*****************************************'
    print ' std_a = '
    print alpha_inv[0,0]
    print ' std_k = '
    print alpha_inv[1,1]
    print ' covariance = '
    print alpha_inv[0,1]

    return [a, k]


def main():

    # initialize the known data
    known_x = np.array([0., 2., 4., 6., 8., 10.,
                        12., 14., 16., 18., 20.])
    known_y = np.array([-0.02802838, 6.107835, 8.233952, 8.526069, 7.438678, 6.297892,
                        5.045212, 3.989115, 3.077397, 2.355075, 1.582216])
    known_sigma = np.array([0.0025, 0.095, 0.086, 0.175, 0.064, 0.045,
                              0.033, 0.041, 0.053, 0.030, 0.025])
    a = 3.1
    k = 0.4

    # do the fitting
    print '-----------------------------------------'
    print '     Nonlinear Chi^2 Fitting Trial 1     '
    print '-----------------------------------------'
    print ' initial a = '
    print a
    print ' initial k = '
    print k
    sol = nonlinear_fitting(known_x, known_y, known_sigma, a, k)

    print '*****************************************'
    print '         Fitting Parameters Found        '
    print '*****************************************'
    print ' a = '
    print sol[0]
    print ' b = '
    print sol[1]

    # plot solution to check validity
    x_range = np.arange(0., 20., .01)
    fx = sol[0]*x_range*np.exp(-sol[1]*x_range)
    plt.figure(1)
    plt.plot(known_x, known_y, 'or', label='Data')  # f(x)
    plt.plot(x_range, fx, label='Trial 1')  # estimated root
    plt.xlabel('$n$')
    plt.ylabel('$e_n$')
    plt.title('Data and $\chi^2$ fitting')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework5_problem2_plot1.png')

    # Show difference between modelling and fitting
    print '*****************************************'
    print '   Differences between Data and Model    '
    print '*****************************************'
    print known_y-(sol[0]*known_x*np.exp(-sol[1]*known_x))

    print '\n\n\n'

    # do the fitting again with new a and k
    a = 300.1
    k = 0.2
    print '-----------------------------------------'
    print '     Nonlinear Chi^2 Fitting Trial 2     '
    print '-----------------------------------------'
    print ' initial a = '
    print a
    print ' initial k = '
    print k
    sol = nonlinear_fitting(known_x, known_y, known_sigma, a, k)

    print '*****************************************'
    print '         Fitting Parameters Found        '
    print '*****************************************'
    print ' a = '
    print sol[0]
    print ' b = '
    print sol[1]

    # plot solution to check validity
    x_range = np.arange(0., 20., .01)
    fx = sol[0]*x_range*np.exp(-sol[1]*x_range)
    plt.figure(1)
    plt.plot(x_range, fx, label='Trial 2')  # estimated root
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework5_problem2_plot1.png')

    # Show difference between modelling and fitting
    print '*****************************************'
    print '   Differences between Data and Model    '
    print '*****************************************'
    print known_y-(sol[0]*known_x*np.exp(-sol[1]*known_x))

    print '\n\n\n'

    # do the fitting again with new fudge factor
    a = 300.1
    k = 0.2
    print '-----------------------------------------'
    print '     Nonlinear Chi^2 Fitting Trial 3     '
    print '-----------------------------------------'
    print ' initial a = '
    print a
    print ' initial k = '
    print k
    sol = nonlinear_fitting(known_x, known_y, known_sigma, a, k, fudge_factor=0.01)

    print '*****************************************'
    print '         Fitting Parameters Found        '
    print '*****************************************'
    print ' a = '
    print sol[0]
    print ' b = '
    print sol[1]

    # plot solution to check validity
    x_range = np.arange(0., 20., .01)
    fx = sol[0]*x_range*np.exp(-sol[1]*x_range)
    plt.figure(1)
    plt.plot(x_range, fx, label='Trial 3')  # estimated root
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework5_problem2_plot1.png')

    # Show difference between modelling and fitting
    print '*****************************************'
    print '   Differences between Data and Model    '
    print '*****************************************'
    print known_y-(sol[0]*known_x*np.exp(-sol[1]*known_x))



if __name__ == "__main__":
    main()