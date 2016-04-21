# Thomas Edwards
# PHYS 5794 - Computational Physics
# 4/2/16
# Homework 9, Problem 1

# Problem statement:

# usage: python phys5794-homework9-problem1.py &

# Imports
import numpy as np
import matplotlib
from random import random as rand
from numpy.random import randint
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


class LatticeMC:

    def __init__(self, l, temp, n0=15., n1=100., total_points=5000):

        self.temp = temp
        self.l = l
        self.lattice = np.zeros((l, l))+1.
        self.init_lattice()

        self.n0 = n0
        self.n1 = n1
        self.total_points = total_points

        # allocate results
        self.ave_abs_m = 0.
        self.m_squared = 0.
        self.m_4 = 0.
        self.total_m = 0
        self.chi = 0.
        self.all_sum = 0.

    def init_lattice(self):
        # initialize in spin down state
        for ix in np.arange(self.l):
            for iy in np.arange(self.l):
                self.lattice[ix, iy] = -1

    def all_mc(self):

        # calculate all MCS
        skip_ix = 0
        while self.total_m <= self.total_points:
            # MCS run
            this_sum = self.this_mc()
            skip_ix += 1
            if skip_ix > self.n1:
                if (skip_ix % self.n0) == 0:
                    self.total_m += 1
                    self.all_sum += this_sum
                    self.ave_abs_m += abs(this_sum/(self.l**2))
                    self.m_squared += (this_sum/(self.l**2))**2
                    self.m_4 += (this_sum/self.l**2)**4
        # finish calculations
        self.ave_abs_m *= 1./self.total_m
        self.m_squared *= 1./self.total_m
        self.m_4 *= 1./self.total_m
        if self.temp > 2.269:
            self.chi = self.m_squared/self.temp*(self.l**2)
        else:
            self.chi = (self.m_squared - self.ave_abs_m**2)*(self.l**2)/self.temp

    def this_mc(self):

        running_sum = 0.
        for ix in np.arange(self.l):
            for iy in np.arange(self.l):
                # get local energy change
                ici = self.lattice[ix, iy]
                ien = self.lattice[(ix+1) % self.l, iy] + self.lattice[(ix-1) % self.l, iy] + self.lattice[ix, (iy+1) % self.l] + self.lattice[ix, (iy-1) % self.l]
                ien = ici*ien
                # check energy change against Metropolis algorithm
                if rand() < self.transition_rate(2.*ien):
                    self.lattice[ix, iy] = -ici
        for ix in np.arange(self.l):
            for iy in np.arange(self.l):
                running_sum += self.lattice[ix, iy]

        return running_sum

    def transition_rate(self, delta_energy):

        # using Metropolis transition rate
        if delta_energy > 0.:
            return np.exp(-delta_energy/self.temp)
        else:
            return 1.

    def get_values(self):
        # return values
        self.all_mc()
        return self.total_m, self.all_sum, self.ave_abs_m, self.m_squared, self.m_4, self.chi

    def auto_corr(self, t, m=10000):
        c1 = 0.
        c2 = 0.
        cn = 0.
        self.samples = np.zeros(m)
        self.autocorr_sums = np.zeros(m)

        for im in np.arange(m):
            self.samples[im] = self.this_mc()

        for ix in np.arange(m-t):
            c1 += (self.samples[ix]*self.samples[ix+t])
        c1 *= (1./(m-t))
        for ix in np.arange(m):
            cn += self.samples[ix]
            c2 += self.samples[ix]**2
        cn = ((1./m)*cn)**2
        c2 *= (1./m)

        return (c1-cn)/(c2-cn)


def main():

    # allocate known values and defaults
    ls = np.array([5, 9, 12])
    temps = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.14, 2.2, 2.35, 2.45, 2.75, 3.0, 3.15, 3.3])
    number_of_ls = ls.shape[0]
    number_of_temps = temps.shape[0]
    avgs = np.zeros([number_of_ls, number_of_temps])
    sqrs = np.zeros([number_of_ls, number_of_temps])
    m4s = np.zeros([number_of_ls, number_of_temps])
    chis = np.zeros([number_of_ls, number_of_temps])
    sums = np.zeros([number_of_ls, number_of_temps])
    total_points = np.zeros([number_of_ls, number_of_temps])

    # do autocorrlation funciton
    autocorr_tries = 50
    autocorr = np.zeros([number_of_ls, autocorr_tries])

    this_l_ix = 0
    for this_l in ls:
        for ix in np.arange(autocorr_tries):
            autocorr[this_l_ix, ix] = LatticeMC(this_l, 3.3).auto_corr(ix)
        this_l_ix += 1

    plt.figure(0)
    plt.plot(autocorr[0, :], label='$L=5$')
    plt.plot(autocorr[1, :], label='$L=9$')
    plt.plot(autocorr[2, :], label='$L=12$')
    plt.title('Autocorrelation Function')
    plt.xlabel('$l$')
    plt.ylabel('$C(l)$')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot0.png')

    it = 0
    il = 0
    for this_l in ls:
        for this_temp in temps:
            total_points[il, it], sums[il, it], avgs[il, it], sqrs[il, it], m4s[il, it], chis[il, it] = LatticeMC(this_l, this_temp).get_values()
            it += 1
        it = 0
        il += 1

    ul = 1.-(m4s/(3.*(sqrs**2)))
    stds = np.sqrt((sqrs-avgs**2.)/(total_points))

    plt.figure(1)
    plt.plot(temps, sqrs[0, :], label='$L=5$')
    plt.plot(temps, sqrs[1, :], label='$L=9$')
    plt.plot(temps, sqrs[2, :], label='$L=12$')
    plt.title('Squared Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Squared Magnetization')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot1.png')

    plt.figure(2)
    plt.plot(temps, avgs[0, :], label='$L=5$')
    plt.plot(temps, avgs[1, :], label='$L=9$')
    plt.plot(temps, avgs[2, :], label='$L=12$')
    plt.title('Average Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot2.png')

    plt.figure(3)
    plt.plot(temps, chis[0, :], label='$L=5$')
    plt.plot(temps, chis[1, :], label='$L=9$')
    plt.plot(temps, chis[2, :], label='$L=12$')
    plt.title('Susceptibility vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot3.png')

    plt.figure(4)
    plt.plot(1./temps, ul[0, :], label='$L=5$')
    plt.plot(1./temps, ul[1, :], label='$L=9$')
    plt.plot(1./temps, ul[2, :], label='$L=12$')
    plt.title('Cumulant vs Inverse Temperature')
    plt.xlabel('$1/T$')
    plt.ylabel('Cumulant')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot4.png')

    plt.figure(5)
    plt.plot(temps, ul[0, :]/ul[1, :], label='$U_5/U_9$')
    plt.plot(temps, ul[1, :]/ul[2, :], label='$U_9/U_{12}$')
    plt.plot(temps, ul[0, :]/ul[2, :], label='$U_5/U_{12}$')
    plt.xlabel('$T$')
    plt.ylabel('Cumulant Ratio')
    plt.title('Cumulant Ratios vs Temperature')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot5.png')

    plt.figure(6)
    plt.plot(temps, stds[0, :], label='$L=5$')
    plt.plot(temps, stds[1, :], label='$L=9$')
    plt.plot(temps, stds[2, :], label='$L=12$')
    plt.xlabel('$T$')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Temperature')
    plt.legend(numpoints=1, loc='upper right')
    plt.savefig('homework9_problem1_plot6.png')


if __name__ == "__main__":
    main()