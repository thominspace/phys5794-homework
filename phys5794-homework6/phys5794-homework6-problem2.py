# Thomas Edwards
# PHYS 5794 - Computational Physics
# 3/14/16
# Homework 6, Problem 2

# Problem statement:

# usage: python phys5794-homework6-problem2.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def shooting_method(delta, eps, y1_at_x0, y2_at_x1, tol=1e-15, iter_cap=1000):

    delta_0 = delta  # y_2(x=0) for f0
    delta_1 = delta + eps  # y_2(x=0) for f1

    for ix in range(iter_cap):

        # determine f0
        x, y1_f0, y2_f0 = rk(0., y1_at_x0, delta_0)
        f0 = y2_f0[-1] - y2_at_x1

        # determine f1
        x, y1_f1, y2_f1 = rk(0., y1_at_x0, delta_1)
        f1 = y2_f1[-1] - y2_at_x1

        error = f1 - f0

        if (abs(error) < tol) or ix==iter_cap-1:
            # within tolerance -> return result
            print 'Ended on iteration ', ix
            return x, y1_f1
        else:
            # secant method update
            delta_2 = delta_1 - ((delta_1 - delta_0)/error)*f1
            delta_0 = delta_1
            delta_1 = delta_2
    

def rk(t_init, y1_init, y2_init):

    # initial conditions
    t_now = 0.
    total_time_steps = 1000
    t_end = 1.
    h = t_end/total_time_steps
    
    t_current = t_init
    y1_current = y1_init
    y2_current = y2_init

    # allocate solution
    t_solved = np.zeros(total_time_steps)
    y1_solved = np.zeros(total_time_steps)
    y2_solved = np.zeros(total_time_steps)

    # track solution index
    solved_index = 0

    while solved_index<total_time_steps:

        # find solution
        t_next, y1_next, y2_next = rk_workhorse(t_current, y1_current, y2_current, h)

        # update solution
        t_solved[solved_index] = t_current
        y1_solved[solved_index] = y1_current
        y2_solved[solved_index] = y2_current

        # progress with next solution by overriding
        y1_current = y1_next
        y2_current = y2_next
        t_current = t_next

        # increase solution index
        solved_index += 1

    return t_solved, y1_solved, y2_solved


def rk_workhorse(t_now, y1_now, y2_now, h):

    # solve y1 portion
    # note that dy1/dt = y2
    y1_k_1 = y2_now
    y1_k_2 = y2_now + (h/2.)*y1_k_1
    y1_k_3 = y2_now + (h/2.)*y1_k_2
    y1_k_4 = y2_now + h*y1_k_3
    y1_next = y1_now + (h/6.)*(y1_k_1 + 2.*y1_k_2 + 2.*y1_k_3 + y1_k_4)

    # now solve the y2 portion
    # dy2/dt = -4pi^2 y(x)
    y2_k_1 = f(y1_now)
    y2_k_2 = f(y1_now+(h/2.)*y1_k_1)
    y2_k_3 = f(y1_now+(h/2.)*y1_k_2)
    y2_k_4 = f(y1_now+h*y1_k_2)
    y2_next = y2_now + (h/6.)*(y2_k_1 + 2.*y2_k_2 + 2.*y2_k_3 + y2_k_4)

    t_next = t_now + h

    return t_next, y1_next, y2_next


def f(y1):

    return -4.*(np.pi**2)*y1


def main():

    # in this problem, dy_1/dx = y_2, and dy_2/dx = -4*pi^2y_1

    y1_at_x0 = 1.
    y2_at_x1 = 2*np.pi

    delta = 0.  # initial guess
    eps = 0.001

    x1, y1 = shooting_method(delta, eps, y1_at_x0, y2_at_x1)

    analytic_solution = np.cos(2*np.pi*x1) + np.sin(2*np.pi*x1)

    plt.figure(1)
    plt.plot(x1, y1, label='$y$')
    plt.plot(x1, analytic_solution, 'r', label='Analytic Soltuion')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Shooting Method Solution')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework6_problem2_plot1.png')

    x2, y2 = shooting_method(delta, eps, y1_at_x0, y2_at_x1, iter_cap=1)

    plt.figure(2)
    plt.plot(x2, y2, label='$y$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Shooting Method Solution, Single Run')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework6_problem2_plot2.png')


if __name__ == "__main__":
    main()