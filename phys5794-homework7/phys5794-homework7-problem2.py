# Thomas Edwards
# PHYS 5794 - Computational Physics
# 3/21/16
# Homework 7, Problem 2

# Problem statement:

# usage: python phys5794-homework7-problem2.py &

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


def explicit(solution_now, h, delta_t, plot_multi, x_space, compare=False):

    solution_next = np.copy(solution_now)
    t_now = 0.
    plot_this = [True, True, True, True, True]

    plt.figure(3*plot_multi)
    if compare:
        plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.0')
    else:
        plt.plot(x_space, solution_next, label='t=0.0')

    while t_now < 0.045:
        t_now += delta_t
        # do the update
        for ix in np.arange(solution_now.shape[0]-2)+1:
            diff = -(solution_now[ix+1] + solution_now[ix-1] - 2.*solution_now[ix])/h**2
            solution_next[ix] = (1.-diff*delta_t)*solution_now[ix]

        if t_now >= 0.005 and plot_this[0]:
            plot_this[0] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.005')
            else:
                plt.plot(x_space, solution_next, label='t=0.005')

        if t_now >= 0.01 and plot_this[1]:
            plot_this[1] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.01')
            else:
                plt.plot(x_space, solution_next, label='t=0.01')

        if t_now >= 0.02 and plot_this[2]:
            plot_this[2] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.02')
            else:
                plt.plot(x_space, solution_next, label='t=0.02')

        if t_now >= 0.03 and plot_this[3]:
            plot_this[3] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.03')
            else:
                plt.plot(x_space, solution_next, label='t=0.03')

        if t_now >= 0.045 and plot_this[4]:
            plot_this[4] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.045')
            else:
                plt.plot(x_space, solution_next, label='t=0.045')

        solution_now = np.copy(solution_next)

    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
    if compare: plt.title('Explicit Solver, Numerical/Analytic Comparison, $\Delta$t='+str(delta_t))
    else: plt.title('Explicit Solver, $\Delta$t='+str(delta_t))
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem2_plot'+str(3*plot_multi)+'.png')


def implicit(solution_now, h, delta_t, plot_multi, x_space, compare=False):

    solution_next = np.copy(solution_now)
    t_now = 0.
    plot_this = [True, True, True, True, True]

    plt.figure(3*plot_multi+1)
    if compare:
        plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.0')
    else:
        plt.plot(x_space, solution_next, label='t=0.0')

    while t_now < 0.045:
        t_now += delta_t

        a_pm = -delta_t/(h**2)
        a_0 = 1.+(2.*delta_t/(h**2))
        end = solution_now.shape[0]-1

        # do the solve
        mat_a = np.zeros((end-1, end-1))
        aminus_diag = np.diag(np.zeros(end-2)+a_pm, k=-1)
        aplus_diag = np.diag(np.zeros(end-2)+a_pm, k=1)
        azero_diag = np.diag(np.zeros(end-1)+a_0, k=0)
        mat_a += aminus_diag + aplus_diag + azero_diag
        mat_b = 2.*np.copy(solution_now[1:end])

        solution_next[1:end] = lusol(mat_a, mat_b) - solution_now[1:end]

        # for ix in np.arange(solution_now.shape[0]-2)+1:
        #     diff = -(solution_now[ix+1] + solution_now[ix-1] - 2.*solution_now[ix])/h**2
        #     solution_next[ix] = (1./(1.+diff*delta_t))*solution_now[ix]

        if t_now >= 0.005 and plot_this[0]:
            plot_this[0] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.005')
            else:
                plt.plot(x_space, solution_next, label='t=0.005')

        if t_now >= 0.01 and plot_this[1]:
            plot_this[1] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.01')
            else:
                plt.plot(x_space, solution_next, label='t=0.01')

        if t_now >= 0.02 and plot_this[2]:
            plot_this[2] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.02')
            else:
                plt.plot(x_space, solution_next, label='t=0.02')

        if t_now >= 0.03 and plot_this[3]:
            plot_this[3] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.03')
            else:
                plt.plot(x_space, solution_next, label='t=0.03')

        if t_now >= 0.045 and plot_this[4]:
            plot_this[4] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.045')
            else:
                plt.plot(x_space, solution_next, label='t=0.045')

        solution_now = np.copy(solution_next)

    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
    if compare: plt.title('Implicit Solver, Numerical/Analytic Comparison, $\Delta$t='+str(delta_t))
    else: plt.title('Implicit Solver, $\Delta$t='+str(delta_t))
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem2_plot'+str(3*plot_multi+1)+'.png')


def sophisticated(solution_now, h, delta_t, plot_multi, x_space, compare=False):

    solution_next = np.copy(solution_now)
    t_now = 0.
    plot_this = [True, True, True, True, True]

    plt.figure(3*plot_multi+2)
    if compare:
        plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.0')
    else:
        plt.plot(x_space, solution_next, label='t=0.0')

    while t_now < 0.045:
        t_now += delta_t

        a_pm = -delta_t/(2.*h**2)
        a_0 = 1.+(2.*delta_t/(2.*h**2))
        end = solution_now.shape[0]-1

        # do the solve
        mat_a = np.zeros((end-1, end-1))
        aminus_diag = np.diag(np.zeros(end-2)+a_pm, k=-1)
        aplus_diag = np.diag(np.zeros(end-2)+a_pm, k=1)
        azero_diag = np.diag(np.zeros(end-1)+a_0, k=0)
        mat_a += aminus_diag + aplus_diag + azero_diag
        mat_b = 2.*np.copy(solution_now[1:end])

        solution_next[1:end] = lusol(mat_a, mat_b) - solution_now[1:end]

        if t_now >= 0.005 and plot_this[0]:
            plot_this[0] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.005')
            else:
                plt.plot(x_space, solution_next, label='t=0.005')

        if t_now >= 0.01 and plot_this[1]:
            plot_this[1] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.01')
            else:
                plt.plot(x_space, solution_next, label='t=0.01')

        if t_now >= 0.02 and plot_this[2]:
            plot_this[2] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.02')
            else:
                plt.plot(x_space, solution_next, label='t=0.02')

        if t_now >= 0.03 and plot_this[3]:
            plot_this[3] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.03')
            else:
                plt.plot(x_space, solution_next, label='t=0.03')

        if t_now >= 0.045 and plot_this[4]:
            plot_this[4] = False
            if compare:
                plt.plot(x_space, solution_next-exact_solution(t_now), label='t=0.045')
            else:
                plt.plot(x_space, solution_next, label='t=0.045')

        solution_now = np.copy(solution_next)

    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
    if compare: plt.title('Sophisticated Solver, Numerical/Analytic Comparison, $\Delta$t='+str(delta_t))
    else: plt.title('Sophisticated Solver, $\Delta$t='+str(delta_t))
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem2_plot'+str(3*plot_multi+2)+'.png')


def exact_solution(t, max_iter=1000):

    x_space = np.linspace(0., 2., 50)
    solution = np.zeros(50)
    # for ix in np.arange(50):
    for n in np.arange(max_iter):
        val = ((2.*n+1.)**2)*(np.pi**2)
        solution += ((1.**n)/val)*np.exp(-((val*t)/4.))*np.sin((n+(1./2.))*np.pi*x_space)
    solution *= 8.
    return solution


def main():

    x_limits = [0., 2.]
    number_of_points = 50.
    x_space = np.linspace(x_limits[0], x_limits[1], number_of_points)
    h = (x_limits[1]-x_limits[0])/number_of_points

    plt.figure(100)
    plt.plot(x_space, exact_solution(0.), label='t=0.0')
    plt.plot(x_space, exact_solution(0.005), label='t=0.005')
    plt.plot(x_space, exact_solution(0.01), label='t=0.01')
    plt.plot(x_space, exact_solution(0.02), label='t=0.02')
    plt.plot(x_space, exact_solution(0.03), label='t=0.03')
    plt.plot(x_space, exact_solution(0.045), label='t=0.045')

    plt.xlabel('$x$')
    plt.ylabel('$\phi$')
    plt.title('Exact Solution')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework7_problem2_plot-1.png')

    initial_psi = np.linspace(x_limits[0], x_limits[1], number_of_points)
    initial_psi[(number_of_points/2)-1:number_of_points] = -initial_psi[(number_of_points/2)-1:number_of_points+1]+2

    # do the solves for our first delta_t
    delta_t = 0.001
    explicit(np.copy(initial_psi), h, delta_t, 0, x_space)
    implicit(np.copy(initial_psi), h, delta_t, 0, x_space)
    sophisticated(np.copy(initial_psi), h, delta_t, 0, x_space)

    # now do comparisons
    explicit(np.copy(initial_psi), h, delta_t, 1, x_space, compare=True)
    implicit(np.copy(initial_psi), h, delta_t, 1, x_space, compare=True)
    sophisticated(np.copy(initial_psi), h, delta_t, 1, x_space, compare=True)

    # solves for second delta_t
    delta_t = 0.00075
    explicit(np.copy(initial_psi), h, delta_t, 2, x_space)
    implicit(np.copy(initial_psi), h, delta_t, 2, x_space)
    sophisticated(np.copy(initial_psi), h, delta_t, 2, x_space)

    # now do comparisons
    explicit(np.copy(initial_psi), h, delta_t, 3, x_space, compare=True)
    implicit(np.copy(initial_psi), h, delta_t, 3, x_space, compare=True)
    sophisticated(np.copy(initial_psi), h, delta_t, 3, x_space, compare=True)


if __name__ == "__main__":
    main()