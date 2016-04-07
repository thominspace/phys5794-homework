# Thomas Edwards
# PHYS 5794 - Computational Physics
# 3/21/16
# Homework 7, Problem 1

# Problem statement:

# usage: python phys5794-homework7-problem1.py &

# Imports
import numpy as np
import time
import matplotlib
from scipy.integrate import quad
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def generate_guess(h, x_limits, y_limits, phi_0):

    x_grids = ((x_limits[1]-x_limits[0])/h)+1
    y_grids = ((y_limits[1]-y_limits[0])/h)+1

    good_grids = True

    if x_grids-np.round(x_grids) != 0:
        print 'Spacing does not allow for consistent grid cells in x.'
        good_grids = False
    if y_grids-np.round(y_grids) != 0:
        print 'Spacing does not allow for consistent grid cells in y.'
        good_grids = False

    if good_grids:
        guess = np.zeros((x_grids, y_grids)) + phi_0
        for ix in range(guess.shape[0]):
            guess[ix,0] = (ix*h)*(1.-(ix*h))  # setting boundary condition Phi(x,y=0)
        return guess, good_grids
    else:
        return np.zeros((0,0)), good_grids


def eliptic_solve(user_guess, user_h, user_w, tol=10e-13, max_iter=10000):

    tic = time.time()

    solution_old = np.copy(user_guess)
    solution_new = np.copy(user_guess)

    w = user_w
    h = user_h

    x_bound = solution_old.shape[0]
    y_bound = solution_old.shape[1]

    x_space = np.linspace(0., 1., x_bound)

    iteration_number = 0

    while True:

        for ix in np.arange(x_bound-2)+1:
            for iy in np.arange(y_bound-2)+1:

                solution_new[ix, iy] = (1.-w)*solution_old[ix, iy] + \
                               (w/4.)*(solution_old[ix+1, iy] + solution_old[ix-1, iy] +
                                       solution_old[ix, iy+1] + solution_old[ix, iy-1] +
                                       h**2*(x_space[ix]**2*(.5-(x_space[ix]/3.))))

        total_energy = 0.
        for ix in np.arange(x_bound-1):
            for iy in np.arange(y_bound-1):

                total_energy += (((solution_new[ix+1, iy] - solution_new[ix, iy])/h)**2 +
                                ((solution_new[ix, iy+1] - solution_new[ix, iy])/h)**2)
        total_energy *= (h**2/2.)

        iteration_number += 1
        if iteration_number == max_iter: break
        if abs(np.sum(solution_old - solution_new)/(x_bound*y_bound)) < tol: break

        solution_old = np.copy(solution_new)

    print 'Final iteration number: ', iteration_number
    return solution_new, total_energy, time.time()-tic


def analytic_solution(x, y, total_sum_iter=100):

    solution = 0.*np.copy(y)
    for n in np.arange(total_sum_iter)+1:
        c_n, err = quad(fx, 0., 1., args=(n,))
        solution += 2.*c_n*np.sin(n*np.pi*x)*np.exp(-n*np.pi*y)

        # gamma = 2.*(-np.pi*n*np.sin(np.pi*n)-2.*np.cos(np.pi*n)+2)/((np.pi*n)**3)
        # solution += gamma*np.sin(np.pi*n*x)*(np.cosh(np.pi*n*y)-np.sinh(np.pi*n*x))

    return solution


def fx(x, n):

    return x*(1.-x)*np.sin(n*np.pi*x)


def main():

    x_limits = [0., 1.]
    y_limits = [0., 10.]

    w_trials = [.05, .04, .03, .02, .01]
    h_trials = [.5, .25, .125, .1]

    phi_0 = 0.

    # analytic solution
    analytic_h = .1
    plt.figure(0)
    x_range = np.arange(x_limits[0], x_limits[1]+analytic_h, analytic_h)
    y_range = np.arange(y_limits[0], y_limits[1]+analytic_h, analytic_h)
    x_space, y_space = np.meshgrid(x_range, y_range)
    analytic = analytic_solution(x_space, y_space)
    solution_contour = plt.contourf(y_space, x_space, analytic)
    cbar = plt.colorbar(solution_contour)
    plt.xlabel('$y$')
    plt.ylabel('$x$')
    plt.title('Analytic Laplace Solution')
    plt.savefig('homework7_problem1_plot0.png')

    total_energy_analytic = 0.
    for ix in np.arange(analytic.shape[0]-1):
        for iy in np.arange(analytic.shape[1]-1):

            total_energy_analytic += (((analytic[ix+1, iy] - analytic[ix, iy])/analytic_h)**2 +
                            ((analytic[ix, iy+1] - analytic[ix, iy])/analytic_h)**2)

    total_energy_analytic *= (analytic_h**2/2.)

    print '------------------'
    print 'Analytic E (for h=0.1):'
    print total_energy_analytic

    # find the numerical solutions

    plot_index = 0

    energy_h_variation = np.zeros(len(h_trials))
    for ih in range(len(h_trials)):
        energy_w_variation = np.zeros(len(w_trials))
        iteration_time_w_variation = np.zeros(len(w_trials))

        for iw in range(len(w_trials)):

            plot_index += 1
            w = w_trials[iw]
            h = h_trials[ih]
            w_tild = 2.*w/h

            print '------------------'
            print 'w = ', w
            print 'h = ', h

            guess, good_grids = generate_guess(h, x_limits, y_limits, phi_0)

            if good_grids:
                solution, final_energy, total_time = eliptic_solve(guess, h, w_tild)
                print 'Elapsed time: ', total_time
                print 'Final E: ', final_energy
                energy_w_variation[iw] = np.copy(final_energy)
                iteration_time_w_variation[iw] = np.copy(total_time)

                x_range = np.arange(x_limits[0], x_limits[1]+h, h)
                y_range = np.arange(y_limits[0], y_limits[1]+h, h)
                x_space, y_space = np.meshgrid(y_range, x_range)
                plt.figure(plot_index)
                solution_contour = plt.contourf(x_space, y_space, solution)
                cbar = plt.colorbar(solution_contour)
                plt.xlabel('$y$')
                plt.ylabel('$x$')
                plt.title('Laplace Solution, w='+str(w)+' h='+str(h))
                plt.savefig('homework7_problem1_plot'+str(plot_index)+'.png')

        plot_index += 1
        plt.figure(plot_index)
        plt.plot(iteration_time_w_variation, energy_w_variation, 'ro')
        plt.xlabel('Iteration Time')
        plt.ylabel('$E$')
        plt.title('E vs. iteration time, h='+str(h))
        plt.savefig('homework7_problem1_plot'+str(plot_index)+'.png')

        energy_h_variation[ih] = energy_w_variation[0]

    plot_index += 1
    plt.figure(plot_index)
    plt.plot(h_trials, energy_h_variation, 'ro', label='Data')

    interp_x_space = np.linspace(0., 0.6, 100)
    poly = np.polyfit(h_trials, energy_h_variation, deg=3)
    plt.plot(interp_x_space, np.polyval(poly, interp_x_space), label='Interpolation')

    plt.xlabel('$h$')
    plt.ylabel('$E$')
    plt.title('E vs. h')
    plt.savefig('homework7_problem1_plot'+str(plot_index)+'.png')

    print '------------------'
    print 'Extrapolated E(h->0): ', np.polyval(poly, 0.)


if __name__ == "__main__":
    main()