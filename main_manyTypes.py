"""
This script performs the stochastic simulations for many types and can also generate Fig4 and Fig5

see main_twoTypes for more comments
"""


import numpy as np
import sys
import timeit
import os
import random
import math
from some_transitionfunctions_manyTypes import *  # transition probabilities and update functions
import itertools
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec

start = timeit.default_timer()
method = 'logistic_Lotka_Volterra'

# gillespie_moran               -- Moran. (2D) this one takes a very long time (attractive fixed point)
# song_JTB                      -- SCPS (4D)
# logistic_Lotka_Volterra       -- logIR (4D)

plot_data = 1  # 1 yes, 0 no

## these options change all other options (see below!)
figure_4 = False
figure_5 = True

####################################################################################################################################
## options ## 
run_until_only_one = 1 # otherwise there is an emergency_t_max
mutate = 1
introduce_dead_parasite = 0


plot_stacked_diversity = 1
plot_simulation = 1  # plot simulation
plot_ODE = 0  # plot integrated differential equation
plot_SDE = 0  # plot stochastic differential equation
plot_fig2 = 0
plot_relativeAbundances = 0
plot_diversity = 1

log = 1  # log for diversity plot

## parameters ##
mutation_rate_host     = 0.005
mutation_rate_parasite = 0.01
time_intro_dead = 16
live_index = 1  # written index-1 (here we start with 0)
kill_index = 4  # written index-1 (here we start with 0)

num_types = 5
delta_t_sde = 0.01 #if method != 'moran' and method != 'local_update' else 1  # time step in SDE integration

colour_h = '#126180' 
colour_p = '#6c3082' 

###### LV #######
if method == 'logistic_Lotka_Volterra':

    birth_H = 6. # 2
    death_P = 1.
    carrying_capacity_LV = 600 #400  # 200 or default:500 = lam0*N_H/num_types = 160
    mu     = birth_H / carrying_capacity_LV  # competition rate is birth rate scales with carrying capacity
    # reactions with two substrates are scaled 
    # (second reactant number would increase rate. thus we take relative number. relative to volume? here we use carrying capacity)
    lam0   = 2*num_types  # base infection rate
    lam    = lam0 / carrying_capacity_LV  # infection rate scaled with carrying capacity 


    h_star = death_P / lam # = dp * K / lam0 = 1 * 500 / 4 = 125 ==> N_H = 250   or 1 * 200/4 = 50 
    p_star = birth_H / lam * (1 - num_types * (h_star) * (mu / birth_H))  # = b * 200/4 * (1-2*50 * 1/200) = b * 25

    N_H = num_types*h_star
    N_P = num_types*p_star
    print('h*', h_star)

    #################

else:
    w_H = 1  # selection intensity host in [0,1]
    w_P = 1 
    N_H = 200  # total host population
    N_P = 200
emergency_t_max = 5000000  # end simulation at 5 000 000
emergency_t_max = 100

np.random.seed(3)  # 1 or 3 or 8/9? good for final figure with logII mu_H=0.005 mu_P=0.01, Bh=6 dP=1 K=600 lam0=10=2*numtypes
random.seed(1)
cMap_h = plt.get_cmap('viridis')
cMap_p = plt.get_cmap('magma')
colours_h = cm.winter(np.linspace(0, 1, num_types))
colours_p = cm.autumn(np.linspace(0, 1, num_types))
indices_h = np.linspace(0, cMap_h.N-90, num_types)  # N is usually 256
indices_p = np.linspace(90, cMap_p.N-1, num_types)
cmap_h = [cMap_h(int(i)) for i in indices_h]
cmap_p = [cMap_p(int(i)) for i in indices_p]
random.shuffle(cmap_h)
random.shuffle(cmap_p)


####################################################################################################################################
## FIGURE 4 ## (introduce P2) 
if figure_4:
    emergency_t_max = 25
    plot_data == 1
    method = 'gillespie_moran'
    np.random.seed(10)  # 10 !! 
    N_H = 200
    N_P = 200
    alpha = 1
    beta = 0
    w_H = 1
    w_P = 1
    num_types = 20
    mutate = 0

    plot_stacked_diversity = 0
    plot_fig2 = 0
    plot_relativeAbundances = 0
    plot_diversity = 1

    log = 1  # log for diversity plot
    run_until_only_one = 1
    time_intro_dead = 16  # manual intro of extinct parasite type
    mutation_rate_host     = 0
    mutation_rate_parasite = 0
    introduce_dead_parasite = 1
    live_index = 1 # (2)  # written_index-1 (here we start with 0)
    kill_index = 4 # (5) # written_index-1 (here we start with 0)
    cMap_h = plt.get_cmap('viridis')
    cMap_p = plt.get_cmap('hot')
    colours_h = cm.winter(np.linspace(0, 1, num_types))
    colours_p = cm.autumn(np.linspace(0, 1, num_types))
    indices_h = np.linspace(0, cMap_h.N, num_types)  # N is usually 256
    indices_p = np.linspace(0, cMap_p.N, num_types)
    cmap_h = [cMap_h(int(i)) for i in indices_h]
    cmap_p = [cMap_p(int(i)) for i in indices_p]

## FIGURE 5 ## (mutations) 
if figure_5:
    run_until_only_one = 0
    mutate = 1
    emergency_t_max = 100
    method = 'logistic_Lotka_Volterra'
    introduce_dead_parasite = 0

    plot_stacked_diversity = 1
    plot_simulation = 1  # plot simulation
    plot_ODE = 0  # plot integrated differential equation
    plot_SDE = 0  # plot stochastic differential equation
    plot_fig2 = 0
    plot_relativeAbundances = 0
    plot_diversity = 0
    plot_data = 1
    plot_total = 1

    log = 1  # log for diversity plot
    mutation_rate_host     = 0.005
    mutation_rate_parasite = 0.01
    num_types = 5
    delta_t_sde = 0.01 #if method != 'moran' and method != 'local_update' else 1  # time step in SDE integration

####################################################################################################################################
if plot_data == 0:
    plot_simulation = 0
    plot_ODE = 0
    plot_SDE = 0

print('N_H', N_H)
print('N_P', N_P)
N = N_H + N_P  # total number of all species

#### Define game

nh_0 = int(N_H / num_types)  # starting population for host
np_0 = int(N_P / num_types)  # starting population for parasite

introduction = 0  # no parasite introduced yet 
alpha = 1  # impact of matching type (p1 (p2) infects h1 (h2))
beta  = 0  # impact of cross-infection (p1 (p2) infects h2 (h1))
Mp = beta  * np.ones((num_types, num_types)) + (alpha - beta) * np.identity(num_types) # impact of parasite on host Mh = - Mp ^ T + alpha + beta
Mh = - Mp.T + beta + alpha  # impact of host on parasite

####################################################################################################################################
#### plot options

if plot_data == 1:

    if plot_stacked_diversity == 1 and plot_total:
        fig_stacked = plt.figure(figsize=(14, 3.5)) 
        gs = gridspec.GridSpec(3, 1, height_ratios=[50, 50, 25]) 
        axes_stacked = [plt.subplot(gs[0]),plt.subplot(gs[1])]
        ax_total = plt.subplot(gs[2])
    elif plot_stacked_diversity == 1:
        fig_stacked, axes_stacked = plt.subplots(ncols = 1, nrows = 2, figsize = [11, 7])

    if plot_relativeAbundances == 1 and plot_diversity == 1:
        fig,  axes  = plt.subplots(nrows = 3, ncols = 1, figsize = [11, 9]) 
        ax_rel = axes[1]
        ax_diversity = axes[2]
        ax_diversity.set_xlabel(r'time')

        ax = axes[0]
    elif plot_relativeAbundances == 1:
        fig,  axes  = plt.subplots(nrows = 2, ncols = 1, figsize = [11, 7]) 
        ax_rel = axes[1]
        ax = axes[0]
    elif plot_diversity == 1:
        # fig,  axes  = plt.subplots(nrows = 2, ncols = 1, figsize = [11, 7]) 
        fig = plt.figure(figsize=(14, 3.5)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[61, 38]) 
        ax = plt.subplot(gs[0])
        ax_diversity = plt.subplot(gs[1])
        ax_diversity.set_xlabel(r'time')

        # ax_diversity = axes[1]
        # ax = axes[0]
    else:
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = [9, 7])
        ax = axes
    if plot_fig2 == 1:
        fig2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize = [9, 7]) 

####################################################################################################################################
############################################### START SIMULATION ###################################################################
####################################################################################################################################
if (plot_simulation == 1 and plot_data == 1):
    start_simulation = timeit.default_timer()
    time = np.array([0])
    all_delta_ts = np.array([])
    total_number_of_steps = 0
    alive = 1
    x = np.array([nh_0 for i in range(num_types)]).reshape((1,num_types))  # all values in time
    y = np.array([np_0 for i in range (num_types)]).reshape((1,num_types))
    n_h = np.array([nh_0 for i in range(num_types)]).astype(int)
    n_h[-1] = N_H - sum(n_h[:-1])
    n_p = np.array([np_0 for i in range(num_types)]).astype(int)
    n_p[-1] = N_P - sum(n_p[:-1])
    types_alive_H = [num_types]
    types_alive_P = [num_types]
    end = False

    while time[-1]<emergency_t_max:  # run as long as there is no extincion
        if run_until_only_one == 1 and alive != 1:
            break
        # compare transition rates in a gillespie algorithm
        # then update new values of h an p

        if method == 'gillespie_moran':
            Th_plus_minus, Tp_plus_minus, T_mutation = transitions(n_h, n_p, N_H, N_P, Mh, Mp, w_H, w_P, mutation_rate_host, mutation_rate_parasite)
            # print('nh', n_h, 'np',  n_p)
            # print(Th_plus_minus, T_mutation)
            try:
                delta_h, delta_p, delta_t = update_gillespie_moran(Th_plus_minus, Tp_plus_minus, T_mutation, num_types, mutate)

                # for i in range(num_types):
                    # if (delta_p[i]>0 and n_h[i]==0):
                        # print(r'$P_{}$'.format({i+1})+' growing but '+r'$H_{}$'.format({i+1})+' dead') 
                        # print(Tp_plus_minus)
            except ValueError:
                print("all probabilities are zero. ")
                print("n_h = ",n_h)
                print("n_p = ",n_p)
                end = True

        elif method == 'song_JTB':
            Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation = current_transitions_4d_song(n_h, n_p, w_H, w_P, Mh, Mp, num_types, mutation_rate_host, mutation_rate_parasite)
            try:
                delta_h, delta_p, delta_t = update_4d(Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation, num_types, mutate)

            except ValueError:
                print("all probabilities are zero. ")
                print("n_h = ",n_h)
                print("n_p = ",n_p)
                current_transitions_4d_song(n_h, n_p, w_H, w_P, Mh, Mp, num_types)
                end = True

        elif method == 'logistic_Lotka_Volterra':
            Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation = current_transitions_4d_LV(n_h, n_p, birth_H, death_P, mu, lam, mutation_rate_host, mutation_rate_parasite)

            try:
                delta_h, delta_p, delta_t = update_4d(Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation, num_types, mutate)

            except ValueError:
                print("all probabilities are zero. ")
                print("n_h = ",n_h)
                print("n_p = ",n_p)
                current_transitions_4d_LV(n_h, n_p, birth_H, death_P, mu, lam, mutation_rate_host, mutation_rate_parasite)
                end = True
        n_p = n_p + delta_p   # update all values
        n_h = n_h + delta_h

        #################################################### INTRO DEAD PARASITE #######################################
        if time[-1] > time_intro_dead and introduction != 1 and introduce_dead_parasite == 1:
            if n_p[live_index]>0:
                print('choose different live_index, this one is still alive')
            else:
                n_p[live_index] += 1
            if n_p[kill_index]>0:
                n_p[kill_index] += -1
            else:
                print('choose different kill_index, this one is already dead')
                quit()
            introduction = 1

        # if np.inner(n_h, n_p) == 0:
            # print("no more matching types")
            # end = True
        types_alive_H_current = sum([i>0 for i in n_h])
        types_alive_P_current = sum([i>0 for i in n_p])

        types_alive_H = np.append(types_alive_H, types_alive_H_current)
        types_alive_P = np.append(types_alive_P, types_alive_P_current)

        # if sum([i == 0 for i in n_h]) == num_types - 1 or sum([i == 0 for i in n_p]) == num_types - 1 or end == True:  # check whether all types are dead
            # alive = 0
            # print("nh", n_h)
            # print("np", n_p)

        if plot_data == 1: # append values to plot
            x = np.append(x, n_h.reshape((1, num_types)), axis = 0)
            y = np.append(y, n_p.reshape((1, num_types)), axis = 0)

        time = np.append(time, time[-1]+delta_t) # append time
        all_delta_ts = np.append(all_delta_ts, delta_t)
        total_number_of_steps += 1

    print("  time: ", time[-1], " alive:", alive)  # print results of this run
    end_simulation = timeit.default_timer()
    print('runtime_simulation', end_simulation - start_simulation)

    print('avg deltat', np.mean(all_delta_ts))
    print('total Number of steps', total_number_of_steps)
    
####################################################################################################################################
#### plot result of simulation
if plot_data == 1 and plot_simulation == 1:

    #####################################
    # figure 1. values in time (trajectories)
    x_total = np.sum(x, axis = 1)
    y_total = np.sum(y, axis = 1)
    tsteps = len(time)
    threshold_legend = int(0.8 * tsteps)
    threshold_legend = int(16./25 * tsteps)

    for i in range(num_types):
        if x[threshold_legend, i] > 0 and num_types > 5:
            ax.plot(time, x[:,i], color = cmap_h[i])#, label = r'$H_{}$'.format({i+1})) 
            ax.scatter([], [], color = cmap_h[i], label = r'$H_{}$'.format({i+1})) 
        else:       
            ax.plot(time, x[:,i], color = cmap_h[i]) 
        # ax.plot(time, x_total, color = 'grey')
    for i in range(num_types):
        if y[threshold_legend, i] > 0 and num_types > 5:
            ax.plot(time, y[:,i], color = cmap_p[i])#, label = r'$P_{}$'.format({i+1})) 
            ax.scatter([], [], color = cmap_p[i], label = r'$P_{}$'.format({i+1})) 
        else:
            ax.plot(time, y[:,i], color = cmap_p[i]) 
        # ax.plot(time, y_total, color = 'grey')

    if num_types<6:
        for i in range(num_types):
            ax.scatter([], [], color = cmap_h[i], label = r'$H_{}$'.format({i+1}))
        for i in range(num_types):
            ax.scatter([], [], color = cmap_p[i], label = r'$P_{}$'.format({i+1}))
        ax.legend()
    
    if plot_relativeAbundances == 1:
        xrel = np.array([i / total for i, total in zip(x, x_total)])
        yrel = np.array([i / total for i, total in zip(y, y_total)])
        for i in range(num_types):
            ax_rel.plot(time, xrel[:,i], color = cmap_h[i], linestyle = '-')
            ax_rel.plot(time, yrel[:,i], color = cmap_p[i], linestyle = '-')

    if plot_diversity == 1:
        ax_diversity.plot(time, types_alive_H, color = colour_h)#, label = r'$n_H$  ')
        ax_diversity.plot(time, types_alive_P, color = colour_p)#,   label = r'$n_P$  ')
        # ax_diversity.scatter([], [], color = "DarkCyan", label = r'$n_H$  ')
        # ax_diversity.scatter([], [], color = "Purple",   label = r'$n_P$  ')
        ax_diversity.annotate(r'$n_H$', xy = (20,10), xycoords = 'data', color = colour_h)
        ax_diversity.annotate(r'$n_P$', xy = (18,5), xycoords = 'data', color = colour_p)
        # ax_diversity.set_ylim(bottom = 10**0)
        ax_diversity.set_yscale('log')

    # print('y', y)
    # print('y_total', y_total)
    # yrel = np.array([i / total if total else i for i, total in zip(y, y_total)])
    # quit()
    # print(yrel)
    if plot_stacked_diversity == 1:
        if plot_total:
            ax_total.plot(time, x_total, color = colour_h)
            ax_total.plot(time, y_total, color = colour_p)
            ax_total.annotate(r'$N_H$', xy = (0.7,0.8), xycoords = "axes fraction", color = colour_h)
            ax_total.annotate(r'$N_P$', xy = (0.75,0.7), xycoords = "axes fraction", color = colour_p)
        xrel = np.array([i / total if total else i for i, total in zip(x, x_total)])
        yrel = np.array([i / total if total else i for i, total in zip(y, y_total)])
        x_cumsum = np.cumsum(xrel, axis = 1)
        y_cumsum = np.cumsum(yrel, axis = 1)

        axes_stacked[0].fill_between(time, 0, x_cumsum[:,0], color = cmap_h[0])
        axes_stacked[1].fill_between(time, 0, y_cumsum[:,0], color = cmap_p[0])
        for i in range(num_types-1):
            axes_stacked[0].fill_between(time, x_cumsum[:,i], x_cumsum[:,i+1], color = cmap_h[i+1])
            axes_stacked[1].fill_between(time, y_cumsum[:,i], y_cumsum[:,i+1], color = cmap_p[i+1])
    #     for i in range(num_types-1):
    #         x_alive_until = x_cumsum[:,i]>0
    #         y_alive_until = y_cumsum[:,i]>0
    #         x_time_i = time[x_alive_until]
    #         y_time_i = time[y_alive_until]
    #         x_cumsum_alive = x_cumsum[x_alive_until,i]
    #         y_cumsum_alive = y_cumsum[y_alive_until,i]
    #         axes_stacked[0].plot(x_time_i, x_cumsum_alive, color = cmap_h[i+1])
    #         axes_stacked[1].plot(y_time_i, y_cumsum_alive, color = cmap_p[i+1])
    # #####################################
    # figure 2. host vs parasite 2d plot
    if plot_fig2 == 1:
        ax2.scatter(x, y, c = time, cmap = cmvir, marker = '.', edgecolor = "none")



####################################################################################################################################
### final plot options
if plot_data == 1:

    try:
        ax.set_xlim([0, time[-1]])
    except NameError:   
        ax.set_xlim(left = 0)


    ax.set_ylabel(r'abundance')
    # if method in methods_2d:
        # ax.set_ylim([0, max(N_H, N_P)])
        # ax2.set_xlim([0, N_H])
        # ax2.set_ylim([0, N_P])
    # else:
        # ax.set_ylim([0, 1000])


    # fig.suptitle('\n'+method+'\n'+r'$\alpha={}$'.format(alpha)+r' $\beta={}$'.format(beta)+r' $N_H={}$'.format(N_H)+r' $N_P={}$'.format(N_P)+r' $w_H={}$'.format(w_H)+r' $w_P={}$'.format(w_P)+r' $d_P={}$'.format(death_P)+r' $b_H={}$'.format(birth_H)+r' $\mu={}$'.format(mu)+r' $\lambda={}$'.format(lam)+r' $\lambda_0={}$'.format(lam0)+r' $K={}$'.format(carrying_capacity_LV)+'\n'+r'$h(0)={}$'.format(nh_0)+r' $p(0)={}$'.format(np_0))
    fig.subplots_adjust(right = 0.91, left = 0.06, bottom = 0.07, top = 0.96, hspace = 0.1)
    ax.legend(loc='center left', bbox_to_anchor = (1.01, 0.50), scatterpoints = 1, ncol=1)
    # if plot_diversity:
        # ax_div  ersity.legend(loc='center left', bbox_to_anchor = (1.01, 0.5), numpoints = 1)
    # ax.set_ylim(bottom = 0)
    # ax.get_xaxis().set_ticks([])

    if plot_relativeAbundances == 1:
        # ax_rel.legend(loc='center left', bbox_to_anchor=(1, 0.2))
        # ax_rel.set_ylim([0,1])
        ax_rel.set_yticks(ax_rel.get_yticks()[:-1])#, minor = True)
        # ax_rel.set_yticks(np.arange(num_types), minor = True)
        ax_rel.set_xlim([0, time[-1]])
        ax.get_xaxis().set_ticks([])
        ax_rel.set_ylabel(r'relative abundance')
        if plot_diversity == 1:
            ax_rel.get_xaxis().set_ticks([])

    if plot_diversity == 1:
        ax_diversity.set_ylabel(r'diversity')
        ax_diversity.set_xlim([0, time[-1]])
        ax_diversity.set_ylim([1, 30])
        # ax_diversity.set_yticks(ax_diversity.get_yticks()[:-1])#, minor = True)
        # ax_diversity.set_yticks(np.arange(num_types), minor = True)
        ax.get_xaxis().set_ticks([])

    if plot_stacked_diversity == 1:
        axes_stacked[0].set_xlim([0, time[-1]])
        axes_stacked[1].set_xlim([0, time[-1]])
        axes_stacked[0].set_ylim([0, 1])
        axes_stacked[1].set_ylim([0, 1])
        if plot_total == 1:
            ax_total.set_xlim([0, time[-1]])
            ax_total.set_xlabel(r'time')
            axes_stacked[1].xaxis.set_visible(False) # Hide only x axis
            ax_total.yaxis.set_visible(False) # Hide only x axis
        else:
            axes_stacked[1].set_xlabel(r'time')
        axes_stacked[0].xaxis.set_visible(False) # Hide only x axis
        axes_stacked[0].yaxis.set_visible(False) # Hide only x axis
        axes_stacked[1].yaxis.set_visible(False) # Hide only x axis
        fig_stacked.subplots_adjust(right = 0.99, left = 0.01, bottom = 0.07, top = 0.99, hspace = 0.1)

    ax.set_xlim([0, time[-1]])

    if plot_fig2 == 1:
        ax2.set_ylim(bottom = 0)
        ax2.set_xlim(left = 0)
        if method in methods_4d:
            ax2.set_xlabel(r"$H_1$")
            ax2.set_ylabel(r"$P_1$")
        else:
            ax2.set_xlabel(r"$H$")
            ax2.set_ylabel(r"$P$")
        # ax2.set_aspect(1)

    if figure_4 == 1:
        ax.axvline(16, linestyle = '--', color = cmap_p[live_index])
        ax_diversity.axvline(16, linestyle = '--', color = cmap_p[live_index])
        
        ax.annotate("",xy=(16/25, 1), xycoords='axes fraction',
                    xytext=(16/25, 1.08), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        ax.annotate(r"$P_2$",xy=(16/25, 1), xycoords='axes fraction',
                    xytext=(16/25, 1.09), textcoords='axes fraction', va = 'bottom', ha = 'center')
        
plt.show()


