# Thomas Edwards
# PHYS 5794 - Computational Physics
# 3/14/16
# Homework 6, Problem 1

# Problem statement:

# usage: python phys5794-homework6-problem1.py &

# Imports
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def test_rk(t_init, y1_init, y2_init):

    # initial conditions
    total_time_steps = 1000
    h = (1./total_time_steps)  # time step
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
        t_next, y1_next, y2_next = test_rk_workhorse(t_current, y1_current, y2_current, h)

        # update solution
        t_solved[solved_index] = t_current
        y1_solved[solved_index] = y1_current
        y2_solved[solved_index] = y2_current

        # progress with next solution by overriding
        t_current = t_next
        y1_current = y1_next
        y2_current = y2_next

        # increase solution index
        solved_index += 1

    return t_solved, y1_solved, y2_solved


def test_rk_workhorse(t_now, y1_now, y2_now, h):

    # solve y1 portion
    # note that dy1/dt = y2
    y1_k_1 = y2_now
    y1_k_2 = y2_now + (h/2.)*y1_k_1
    y1_k_3 = y2_now + (h/2.)*y1_k_2
    y1_k_4 = y2_now + h*y1_k_3
    y1_next = y1_now + (h/6.)*(y1_k_1 + 2.*y1_k_2 + 2.*y1_k_3 + y1_k_4)

    # now solve the y2 portion
    # dy2/dt = -4pi^2 y(x)
    y2_k_1 = test_f(y1_now)
    y2_k_2 = test_f(y1_now+(h/2.)*y1_k_1)
    y2_k_3 = test_f(y1_now+(h/2.)*y1_k_2)
    y2_k_4 = test_f(y1_now+h*y1_k_2)
    y2_next = y2_now + (h/6.)*(y2_k_1 + 2.*y2_k_2 + 2.*y2_k_3 + y2_k_4)

    t_next = t_now + h

    return t_next, y1_next, y2_next


def test_f(y):

    return -4*np.pi**2*y


def rk(t_init, theta_init, w_init, b):

    # initial conditions
    t_now = 0.
    total_time_steps = 1000
    gl = np.sqrt(2./(3.*11.931))
    h = (3.*np.pi)/100.  # time step
    t_elapsed = total_time_steps*h  # elapsed time
    t_current = t_init
    theta_current = theta_init
    w_current = w_init

    # allocate solution
    t_solved = np.zeros(total_time_steps)
    theta_solved = np.zeros(total_time_steps)
    w_solved = np.zeros(total_time_steps)

    # track solution index
    solved_index = 0

    while solved_index<total_time_steps:

        # find solution
        t_next, theta_next, w_next = rk_workhorse(t_current, theta_current, w_current, h, b)

        # update solution
        t_solved[solved_index] = t_current
        theta_solved[solved_index] = theta_current
        w_solved[solved_index] = w_current

        # progress with next solution by overriding
        theta_current = theta_next
        w_current = w_next
        t_current = t_next

        # increase solution index
        solved_index += 1

    return t_solved, theta_solved, w_solved


def rk_workhorse(t_now, theta_now, w_now, h, b):

    # solve theta portion
    # note that dtheta/dt = w
    theta_k_1 = w_now
    theta_k_2 = w_now + (h/2.)*theta_k_1
    theta_k_3 = w_now + (h/2.)*theta_k_2
    theta_k_4 = w_now + h*theta_k_3
    theta_next = theta_now + (h/6.)*(theta_k_1 + 2.*theta_k_2 + 2.*theta_k_3 + theta_k_4)

    # now solve the w portion
    # dw/dt = -sin(theta) - qw + bcos(w_0 t')
    omega_k_1 = f(t_now, theta_now, w_now, b)
    omega_k_2 = f(t_now+(h/2.), theta_now+(h/2.)*theta_k_1, w_now+(h/2.)*omega_k_1, b)
    omega_k_3 = f(t_now+(h/2.), theta_now+(h/2.)*theta_k_2, w_now+(h/2.)*omega_k_2, b)
    omega_k_4 = f(t_now+h, theta_now+h*theta_k_2, w_now+h*omega_k_3, b)
    omega_next = w_now + (h/6.)*(omega_k_1 + 2.*omega_k_2 + 2.*omega_k_3 + omega_k_4)

    t_next = t_now + h

    return t_next, theta_next, omega_next


def f(t, theta, w, b):

    w_0bar = (2./3.)
    q = 0.5
    return -np.sin(theta) - q*w + b*np.cos(w_0bar*t)


def main():

    # verification case
    # let's just solve d^2y/dt^2 = -4pi^2x for y(0)=1, y'(0)+0
    test_t, test_y1, test_y2 = test_rk(0., 1., 0.)

    plt.figure(10)
    plt.plot(test_t, test_y1, 'xr', label='Position')  # theta vs w
    plt.plot(test_t, np.cos(2*np.pi*test_t), 'b', label='Known Solution')
    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.title('Phase Diagram for Testing')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework6_problem1_plot10.png')

    # problem case

    w_0 = 2.0  # modified for t_prime
    theta_0 = 0.

    f_d0 = np.array([0., 0.89, 1.145]) # note that this is in units on mg, so b = f_d0 here, not f_d0/mg
    for ix in range(len(f_d0)):
        b = f_d0[ix]
        t, theta, w = rk(0., theta_0, w_0, b)
        t *= (2./(3.*11.931))  # converting back to t, from t_prime

        # plt.figure(ix*2)
        # plt.plot(t, theta, 'or', label='$\\theta$')  # theta
        # plt.plot(t, w, 'ob', label='$\omega$')  # w
        # plt.xlabel('$t$')
        # plt.ylabel('$\\theta$, $\omega$')
        # plt.title('$t$ vs. $\\theta$, $\omega$')
        # plt.legend(numpoints=1, loc='upper right')
        # plt.savefig('homework6_problem1_plot'+str(ix*2)+'.png')

        plt.figure(ix*2 + 1)
        plt.plot(theta, w, 'xr', label='Position')  # theta vs w
        plt.xlabel('$\\theta$')
        plt.ylabel('$\omega$')
        plt.title('Phase Diagram')
        plt.legend(numpoints=1, loc='upper right')
        plt.savefig('homework6_problem1_plot'+str(ix*2 + 1)+'.png')

if __name__ == "__main__":
    main()