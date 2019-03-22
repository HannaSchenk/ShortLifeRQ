"""
This script performs the stochastic simulations for two types and can also generate Fig1

This script numerically analyses different methods to model host-parasite interactions with more or less constant or variable population
size. The population size is always meant as the number of hosts N_H of which there are two types OR the number of parasites N_P of which 
there are two types. It is not meaningful to talk about the population size of both species together. In a model with strict constant 
population size the dynamics are thus 2-dimensional (the amount of second type within a species follows from the abundance of the first type
and the population size), whereas the other models are 4-dimensionl.

2d models:
 -- The classical moran process is a discrete-time birth-death process where birth events depend on the relative fitness of an individual to 
    the average fitness of the population. 
 -- The local_update process is a variation, where fitness is measured locally (against a second type) rather than globally (against an average).
 -- Both processes can be implemented in continuous time with a gillespie algorithm. 
 4d models:
 -- Antagonistic interactions can also be modeled with the traditional Lotka-Volterra model (independent reactions IR).
     We use a version with logistic growth (logIR) to get a handle on population size via a carrying capacity. This allows for a changing, 
    yet slightly constrained overall population size
 -- It is also possible to use a classical Lotka-Volterra model but adapt the death rate of the prey/host and the birthrate of the 
    predator/parasite dynamically such that average birthrates equal average deathrates for each species. This is what was achieved in song_JTB (SCPS)

Please define (1) what methods to loop through and (2) the birth rates (which defines NH and NP in the end)

This script can plot or save values obtained by 
 (i)   simulating via gillespie algorithm 
 (ii)  SDE integration
 (iii) ODE integration (deterministic model)

"""

import numpy as np
import sys
import timeit
import os
import random
import math
from all_transition_functions import *  # transition probabilities and update functions
import matplotlib.cm as cm

# -----------------------------------------------------------------------------------------------------------------------------------
#### OPTIONS
# -----------------------------------------------------------------------------------------------------------------------------------

start = timeit.default_timer()
figure1 = False             # figure 1 in Paper
plot_fig2 = True            # plot h1 vs p1 trajectory (and h2 vs p2 if 4D method)
plot_simulation = True      
save_simulation = False     # save extinction times to file
plot_SDE = False             # plot stochastic simulation via SDE 
save_SDE = False            # save extinction times by SDE to file
plot_ODE = False            # plot integrated differential equation

runs = 1                    # how many runs - integer

# runs = int(sys.argv[2])

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
#### METHOD
# -----------------------------------------------------------------------------------------------------------------------------------

# local_update                  -- dtPC         pairwise comparison (discrete time)
# moran                         -- dtMoran      (discrete time)
# gillespie_local_update        -- PC
# gillespie_moran               -- Moran        this one takes a very long time (attractive fixed point)
# song_JTB                      -- SCPS         (4D)
# logistic_Lotka_Volterra       -- logIR        (4D)
# Lotka_Volterra                -- IR           (4D) mu = 0, no host competition

sde_method = 'Runge_Kutta'  # Euler or Runge_Kutta

methods_2d = ['local_update', 'moran', 'gillespie_local_update', 'gillespie_moran']
methods_4d = ['song_JTB', 'logistic_Lotka_Volterra']

methods    = ['local_update', 'moran', 'gillespie_local_update', 'gillespie_moran', 'song_JTB', 'logistic_Lotka_Volterra', 'Lotka_Volterra']
methods    = ['gillespie_local_update', 'song_JTB']
# methods = [str(sys.argv[1])]



# -----------------------------------------------------------------------------------------------------------------------------------
#### parameters
delta_t_sde = 0.01 if (method != 'moran' and method != 'local_update') else 0.1  # time step in SDE integration
all_birth_H = [0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2, 1.28, 1.36, 1.44, 1.52, 1.6]  # or half of this when mu=0 in non-logistic "Lotka_Volterra"
all_birth_H = [0.24, 0.32]

if figure1:
    all_birth_H = [6]
    seed_value = 10
    np.random.seed(seed_value)

death_P = 1.
carrying_capacity_LV = 500                                          # 200 or default:500

# -----------------------------------------------------------------------------------------------------------------------------------
for birth_H in all_birth_H:  # LOOP THROUGH all_birth_H. One figure for each
    mu     = birth_H / carrying_capacity_LV                         # competition rate is birth rate scaled with carrying capacity
    lam0   = 4.                                                     # base infection rate
    lam    = lam0 / carrying_capacity_LV                            # infection rate scaled with carrying capacity 

    h_star = death_P / lam                                          # = dp * K / lam0 = 1 * 500 / 4 = 125 ==> N_H = 250   or 1 * 200/4 = 50 
    p_star = birth_H / lam * (1 - 2 * (h_star) * (mu / birth_H))    # = b * 200/4 * (1-2*50 * 1/200) = b * 25

    # -----------------------------------------------------------------------------------------------------------------------------------
    if figure1:
        plot_simulation = True  
        save_simulation = False
        plot_SDE = False
        save_SDE = False
        runs    = 1
        methods = ['gillespie_moran', 'logistic_Lotka_Volterra']
        birth_H = 6
        lam0    = 4
        carrying_capacity_LV = 100  
        lam     = lam0 / carrying_capacity_LV        # infection rate scaled with carrying capacity 
        mu      = birth_H / carrying_capacity_LV     # competition rate is birth rate scales with carrying capacity
        h_star  = death_P / lam                      # = dp * K / lam0 = 1 * 500 / 4 = 125 ==> N_H = 250   or 1 * 200/4 = 50 
        p_star  = birth_H / lam * (1 - 2 * (h_star) * (mu / birth_H))  # = b * 200/4 * (1-2*50 * 1/200) = b * 25

    # -----------------------------------------------------------------------------------------------------------------------------------

    print(method, 'h*', h_star, 'p*', p_star)

    N_H = round(2 * h_star)  # total host population
    N_P = round(2 * p_star)

    print(r'$N_H=$', N_H, r' $N_P=$', N_P)


    h0_ode = np.array([N_H / 2 + 0.3 * N_H]) / N_H  # slightly off the fixed point for ODE...
    p0_ode = np.array([N_P / 2]) / N_P
    h0_sde = np.array([N_H / 2]) / N_H  # ...not necessary here because diffusion destabilises
    p0_sde = np.array([N_P / 2]) / N_P

    N = N_H + N_P  # total number of all species

    w_H = 0.5
    w_P = 1.

    all_colours_h = cm.winter(np.linspace(0, 1, len(methods)*2))
    all_colours_p = cm.spring(np.linspace(0, 1, len(methods)*2))

    emergency_t_max = 5000000  # end simulation
    # -----------------------------------------------------------------------------------------------------------------------------------

    ####################################################################################################################################
    #### plot options

    colour_Nh = '#126180'
    colour_Np = '#6c3082' 
    colours_h = ['#267f53', '#70d4a2', '#708ed4']
    colours_p = ['#c44082', '#d470a2', '#d47a70'] 

    if plot_simulation or plot_SDE or plot_ODE:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fs = 14

        colors = cm.viridis(np.linspace(0, 1, runs))
        fig,  all_ax  = plt.subplots(nrows=2, ncols=len(methods), figsize=[12,6])

        if plot_fig2:
            fig2, ax2 = plt.subplots(nrows=1, ncols=1)

    # -----------------------------------------------------------------------------------------------------------------------------------
    for index_method, method in enumerate(methods):   ### LOOP THROUGH METHODS
        print('method: ', method)
        if plot_simulation or plot_SDE or plot_ODE:
            ax     = all_ax[0,index_method]  # plot trajectories in time
            ax_rel = all_ax[1,index_method]  # plot p1 against h1


        print('w_H', w_H)
        print('w_P', w_P)

        if method == 'Lotka_Volterra':
            method = 'logistic_Lotka_Volterra'
            mu = 0                                 # set competition to 0 
            birth_H = birth_H/2.                   # rescale
            methods
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        #### Define game

        alpha = 1                                           # impact of matching type (p1 (p2) infects h1 (h2))
        beta  = 0                                           # impact of cross-infection (p1 (p2) infects h2 (h1))
        Mh    = np.array([[beta, alpha], [alpha, beta]])    # impact of parasite on host Mh = - Mp ^ T + alpha + beta
        Mp    = np.array([[alpha, beta], [beta, alpha]])    # impact of host on parasite

        if plot_simulation or plot_SDE or plot_ODE:

            if runs == 1:
                x = [int(N_H / 2)]
                y = [int(N_P / 2)]
                cmvir = plt.cm.get_cmap('viridis')

        # for moran and local update (2d methods) we can precompute the transition probabilities 
        # because the population size stays constant the transition matrices are constant
        # the function returns 'nan' for methods song_JTB and logistic_Lotka_Volterra
        tr = transitions(method, N_H, N_P, Mh, Mp, w_H, w_P)  # transition probabilities - see 'all_transition_functions.py'
        # -----------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- START SIMULATION ---------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------
        if plot_simulation or save_simulation:

            run = 0

            while run<runs:  # loop through runs
                start_simulation = timeit.default_timer()

                nh_0 = int(N_H / 2)             # starting population for host
                np_0 = int(N_P / 2)             # starting population for parasite

                if method in methods_4d:        # 4D requires more initial conditions
                    nh_02 = int(N_H / 2)
                    np_02 = int(N_P / 2)
                    if plot_ODE and method == 'logistic_Lotka_Volterra':
                        h0_ode  = h0_ode * N_H
                        p0_ode  = p0_ode * N_P
                        h20_ode = np.array([N_H / 2 - 0.3 * N_H])
                        p20_ode = np.array([N_P / 2])
                    if plot_SDE or save_SDE:
                        h0_sde  = h0_sde * N_H
                        p0_sde  = p0_sde * N_P
                        h20_sde = np.array([N_H / 2])
                        p20_sde = np.array([N_P / 2])

                # start each run in the same initial condition
                n_h = nh_0
                n_p = np_0
                time = np.array([0])
                all_delta_ts = np.array([])
                total_number_of_steps = 0
                alive = 1

                if method in methods_4d:  # 4d requires more variables
                    n_h2 = nh_02
                    n_p2 = np_0
                    x2 = [nh_02]
                    y2 = [np_0]

                while alive == 1 and time[-1]<emergency_t_max:  # run as long as there is no extincion

                    # use precomputed transition rates/probabilities or compute current 4d rates
                    # and compare these probabilities with a random number (for moran and local_update) or use the rates in a gillespie algorithm
                    # then update new values of h an p

                    if method in methods_2d:
                        current_probabilities = tr[:, n_h, n_p]
                        delta_h, delta_p, delta_t = update(method, current_probabilities)  # this is the actual update where a (pseudo-)random number is used

                        n_p += delta_p  # update all values
                        n_h += delta_h

                        if any([n_h == 0, n_h == N_H, n_p == 0, n_p == N_P]):  # check whether all types are alive
                            alive = 0

                    elif method in methods_4d:
                        if method == 'song_JTB':
                            current_probabilities = current_transitions_4d_song(n_h, n_h2, n_p, n_p2, w_H, w_P, alpha, beta)
                        else:
                            current_probabilities = current_transitions_4d_LV(n_h, n_h2, n_p, n_p2, birth_H, death_P, mu, lam)
                        
                        delta_h, delta_h2, delta_p, delta_p2, delta_t = update_4d(current_probabilities)  # this is the actual update where a (pseudo-)random number is used

                        n_p  += delta_p   # update all values
                        n_h  += delta_h
                        n_p2 += delta_p2
                        n_h2 += delta_h2
                        
                        if any([n_h2 == 0, n_h == 0, n_p2 == 0, n_p == 0]):  # check whether all types are alive
                            alive = 0
                    else:
                        print('wrong method')
                        quit()

                    if plot_simulation and runs == 1: # append values to plot
                        x.append(n_h)
                        y.append(n_p)
                        if method in methods_4d:
                            x2.append(n_h2)
                            y2.append(n_p2)

                    time = np.append(time, time[-1]+delta_t) # append time
                    all_delta_ts = np.append(all_delta_ts, delta_t)
                    total_number_of_steps += 1

                #### save data
                if save_simulation:
                    if method == 'logistic_Lotka_Volterra':
                        file_name = './data/'+method+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_mu{:.3f}'.format(mu)+'_lam{:.3f}'.format(lam)+'_lam0{:.3f}'.format(lam0)+'_birthH{:.3f}'.format(birth_H)+'_deathP{:.3f}'.format(death_P)+'_carryingCapLV{:.3f}'.format(carrying_capacity_LV)+'.txt'
                    else:
                        file_name = './data/'+method+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_wH{:.3f}'.format(w_H)+'_wP{:.3f}'.format(w_P)+'_alpha{:d}'.format(alpha)+'_beta{:d}'.format(beta)+'.txt'
                    file_times = open(file_name, 'ab')
            
                    np.savetxt(file_times, [time[-1]]) 
                    file_times.close()

                print("run: ", run, "  end time: ", time[-1], " alive:", alive)  # print results of this run
                run += 1
                end_simulation = timeit.default_timer()
                # -----------------------------------------------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------
        #### plot result of simulation
        if plot_simulation and runs == 1:

            #####################################
            # figure 1. values in time (trajectories)

            if method in methods_2d:  # 2d
                ax.plot(time, x, color = colours_h[0], label = r'$H_1$') 
                ax.plot(time, [N_H-i for i in x], color = colours_h[1], label = r'$H_2$') 
                ax.plot(time, y, color = colours_p[0], label = r'$P_1$') 
                ax.plot(time, [N_P-j for j in y], color = colours_p[1], label = r'$P_2$')
                ax.plot(time, [N_H for i in x], color = colour_Nh, linewidth=2.5) 
                ax.plot(time, [N_P for j in y], color = colour_Np, linewidth=2.5) 
                xrel1 = [i / N_H for i in x] 
                yrel1 = [i / N_P for i in y]
                xrel2 = [1 - i for i in xrel1]
                yrel2 = [1 - i for i in yrel1]
                ax_rel.plot(time, xrel1, color = colours_h[0], linestyle = '-', label = r'$h_1$')
                ax_rel.plot(time, xrel2, color = colours_h[1], linestyle = '-', label = r'$h_2$')
                ax_rel.plot(time, yrel1, color = colours_p[0], linestyle = '-', label = r'$p_1$')
                ax_rel.plot(time, yrel2, color = colours_p[1], linestyle = '-', label = r'$p_2$')
                if plot_fig2:
                    ax2.scatter(x, y, c=time, cmap=cmvir, marker='.')
            elif method in methods_4d:  # 4d
                x_total = [xi + xj for xi, xj in zip(x, x2)]
                y_total = [yi + yj for yi, yj in zip(y, y2)]
                ax.plot(time, x,  color = colours_h[0], linestyle = '-', label = r'$H_1$') 
                ax.plot(time, x2, color = colours_h[1], linestyle = '-', label = r'$H_2$') 
                ax.plot(time, y,  color = colours_p[0], linestyle = '-', label = r'$P_1$') 
                ax.plot(time, y2, color = colours_p[1], linestyle = '-', label = r'$P_2$') 
                ax.plot(time, x_total, color = colour_Nh) 
                ax.plot(time, y_total, color = colour_Np)
                xrel1 = [i / total for i, total in zip(x,  x_total)] 
                yrel1 = [i / total for i, total in zip(y,  y_total)] 
                xrel2 = [i / total for i, total in zip(x2, x_total)]
                yrel2 = [i / total for i, total in zip(y2, y_total)]
                ax_rel.plot(time, xrel1, color = colours_h[0], linestyle = '-', label = r'$h_1$')
                ax_rel.plot(time, xrel2, color = colours_h[1], linestyle = '-', label = r'$h_2$')
                ax_rel.plot(time, yrel1, color = colours_p[0], linestyle = '-', label = r'$p_1$')
                ax_rel.plot(time, yrel2, color = colours_p[1], linestyle = '-', label = r'$p_2$')
                if plot_fig2:
                    ax2.scatter(x, y, c=time, cmap=cmvir, marker='.')
                    ax2.scatter(x2, y2, c=time, cmap=cmvir, marker='.')
                    
        # -----------------------------------------------------------------------------------------------------------------------------------
        #### deterministic numerical integration
        if plot_ODE:
            from all_odes import *  # ode functions

            start_ode = timeit.default_timer()

            try:
                time_ode = np.linspace(0.0, time[-1], 10000)
            except NameError:
                time_ode = np.linspace(0, 500000, 50000)

            if method in ['local_update', 'moran', 'gillespie_local_update', 'gillespie_moran', 'song_JTB']:  # 2d (song_JTB reduces to 2d)
                h_ode, p_ode = integrate_2d(method, h0_ode, p0_ode, time_ode, w_H, w_P, Mh, Mp, N_H, N_P)  
                ax.plot(time_ode, N_H - h_ode, color = colours_h[1])
                ax.plot(time_ode, N_P - p_ode, color = colours_p[1])
            elif method in ['logistic_Lotka_Volterra']:  # 4d
                h_ode, h2_ode, p_ode, p2_ode = integrate_4d_LV(h0_ode, h20_ode, p0_ode, p20_ode, time_ode, birth_H, death_P, mu, lam)
                ax.plot(time_ode, h2_ode, color = colours_h[1])
                ax.plot(time_ode, p2_ode, color = colours_p[1])
                if plot_fig2:
                    ax2.scatter(h2_ode, p2_ode, c = time_ode, cmap = cmvir, marker = '.', edgecolor = 'None')


            ax.plot(time_ode, h_ode, color = colours_h[0])
            ax.plot(time_ode, p_ode, color = colours_p[0])
            if plot_fig2:
                ax2.scatter(h_ode, p_ode, c = time_ode, cmap = plt.cm.get_cmap('plasma'), marker = '.', edgecolor = 'None')
            
            end_ode = timeit.default_timer()

            print("runtime_ode : ", end_ode - start_ode)


        # -----------------------------------------------------------------------------------------------------------------------------------
        #### stochastic differential equation integration
        if plot_SDE or save_SDE:
            from all_sdes import *  # sde functions
            print('dt_sde', delta_t_sde)
            print('sde method', sde_method)
            tmax_sde = emergency_t_max


            if save_SDE:
            
                if method in methods_2d or method == 'song_JTB':
                    file_name = './data_sde/'+method+sde_method+'_dtsde{:.3f}'.format(delta_t_sde)+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_wH{:.3f}'.format(w_H)+'_wP{:.3f}'.format(w_P)+'_alpha{:d}'.format(alpha)+'_beta{:d}'.format(beta)+'.txt'
                elif method == 'logistic_Lotka_Volterra':
                    file_name = './data_sde/'+method+sde_method+'_dtsde{:.3f}'.format(delta_t_sde)+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_mu{:.3f}'.format(mu)+'_lam{:.3f}'.format(lam)+'_lam0{:.3f}'.format(lam0)+'_birthH{:.3f}'.format(birth_H)+'_deathP{:.3f}'.format(death_P)+'_carryingCapLV{:.3f}'.format(carrying_capacity_LV)+'.txt'
                else:
                    print('choose method')
            
                run = 0
                while run<runs:  # loop through runs

                    if method in methods_2d:
                        h_sde, p_sde, t_sde, error_in_sde = stoch_integrate_2d(method, sde_method, tmax_sde, delta_t_sde, np.concatenate([h0_sde, p0_sde]), N_H, N_P, Mh, Mp, w_H, w_P)
                    elif method in ['logistic_Lotka_Volterra']:
                        h_sde, h2_sde, p_sde, p2_sde, t_sde, error_in_sde = stoch_integrate_4d_LV(sde_method, tmax_sde, delta_t_sde, h0_sde, h20_sde, p0_sde, p20_sde, birth_H, death_P, mu, lam)
                    elif method in ['song_JTB']:
                        h_sde, h2_sde, p_sde, p2_sde, t_sde, error_in_sde = stoch_integrate_4d_song(sde_method, tmax_sde, delta_t_sde, np.array([h0_sde, h20_sde, p0_sde, p20_sde]).flatten(), alpha, beta, w_H, w_P)
                    else:
                        print('choose method')

                    if error_in_sde == 0:
                        print('dt_sde: ', delta_t_sde, "   end time sde : ", t_sde[-1])
                        file_times = open(file_name, 'ab')
                        np.savetxt(file_times, [t_sde[-1]]) 
                        file_times.close()
                    else:
                        print('error')
                    run += 1

            # -----------------------------------------------------------------------------------------------------------------------------------
            if plot_SDE:
                start_sde = timeit.default_timer()

                if method in methods_2d:
                    h_sde, p_sde, t_sde, error_in_sde = stoch_integrate_2d(method, sde_method, tmax_sde, delta_t_sde, np.concatenate([h0_sde, p0_sde]), N_H, N_P, Mh, Mp, w_H, w_P)
                    ax.plot(t_sde, h_sde,     linestyle = '--', color = colours_h[0], label = r'$h_1$'+r' SDE')
                    ax.plot(t_sde, N_H-h_sde, linestyle = '--', color = colours_h[1], label = r'$h_2$'+r' SDE')
                    ax.plot(t_sde, p_sde,     linestyle = '--', color = colours_p[0], label = r'$p_1$'+r' SDE')
                    ax.plot(t_sde, N_P-p_sde, linestyle = '--', color = colours_p[1], label = r'$p_2$'+r' SDE')

                elif method in ['logistic_Lotka_Volterra']:
                    h_sde, h2_sde, p_sde, p2_sde, t_sde, error_in_sde = stoch_integrate_4d_LV(sde_method, tmax_sde, delta_t_sde, h0_sde, h20_sde, p0_sde, p20_sde, birth_H, death_P, mu, lam)
                    ax.plot(t_sde, h_sde,  linestyle = '--', dashes = [10,10], color = colours_h[0], label = r'$h_1$'+r' SDE')
                    ax.plot(t_sde, h2_sde, linestyle = '--', dashes = [10,10], color = colours_h[1], label = r'$h_2$'+r' SDE')
                    ax.plot(t_sde, p_sde,  linestyle = '--', dashes = [10,10], color = colours_p[0], label = r'$p_1$'+r' SDE')
                    ax.plot(t_sde, p2_sde, linestyle = '--', dashes = [10,10], color = colours_p[1], label = r'$p_2$'+r' SDE')

                elif method in ['song_JTB']:
                    h_sde, h2_sde, p_sde, p2_sde, t_sde, error_in_sde = stoch_integrate_4d_song(sde_method, tmax_sde, delta_t_sde, np.array([h0_sde, h20_sde, p0_sde, p20_sde]).flatten(), alpha, beta, w_H, w_P)
                    ax.plot(t_sde, h_sde,  linestyle = '--', dashes = [100,100], color = colours_h[0], label = r'$h_1$'+r' SDE')
                    ax.plot(t_sde, h2_sde, linestyle = '--', dashes = [10,10],   color = colours_h[1], label = r'$h_2$'+r' SDE')
                    ax.plot(t_sde, p_sde,  linestyle = '--', dashes = [10,10],   color = colours_p[0], label = r'$p_1$'+r' SDE')
                    ax.plot(t_sde, p2_sde, linestyle = '--', dashes = [10,10],   color = colours_p[1], label = r'$p_2$'+r' SDE')


            end_sde = timeit.default_timer()
            print("end time sde : ", t_sde[-1])
            # print("runtime_sde : ", end_sde - start_sde)

        end = timeit.default_timer()
        # print("runtime : ", end - start)

        # -----------------------------------------------------------------------------------------------------------------------------------
        ### final plot options
        if plot_simulation or plot_SDE or plot_ODE:
            if figure1:
                print('nh', n_h, 'np', n_p)
                if n_h == 0:
                    coldead = colours_h[0]
                elif n_p == 0:
                    coldead = colours_p[0]
                elif n_h2 == 0:
                    coldead = colours_h[1]
                else:
                    coldead = colours_p[1]
                ax.annotate("",xy=(time[-1], 0), xycoords='data',
                            xytext=(time[-1], 17), textcoords='data',
                            arrowprops=dict(headlength=6,headwidth=8,width=2, color = coldead), color = coldead)
                ax_rel.annotate("",xy=(time[-1], 0), xycoords='data',
                                xytext=(time[-1], 17/200), textcoords='data',
                                arrowprops=dict(headlength=6,headwidth=8,width=2, color = coldead), color = coldead)
                ax_rel.set_xlim(ax.get_xlim())
            if index_method == 0:
                ax.set_ylabel(r'abundance', fontsize = fs)
                if figure1:
                    ax.annotate(r'Constant population size', xy = (0.5, 1.045), verticalalignment="center", horizontalalignment="center", xycoords = "axes fraction", fontsize = fs)
                    ax.annotate(r'$P_1$', xy = (0.15, 0.6), xycoords = "axes fraction", color = colours_p[0], fontsize = fs)
                    ax.annotate(r'$P_2$', xy = (0.25, 0.55), xycoords = "axes fraction", color = colours_p[1], fontsize = fs)
            else:
                
                if figure1:
                    ax.annotate(r'Changing population size', xy = (0.5, 1.045), verticalalignment="center", horizontalalignment="center", xycoords = "axes fraction", fontsize = fs)
                    ax.annotate(r'$N_P$', xy = (0.43, 0.75), xycoords = "axes fraction", color = colour_Np, fontsize = fs)
                    ax.annotate(r'$N_H$', xy = (7/9, 1/3), xycoords = "axes fraction", color = colour_Nh, fontsize = fs)
                ax.yaxis.set_visible(False)

                try:
                    ax_rel.yaxis.set_visible(False)
                    if figure1:
                        ax_rel.annotate(r'$H_1$', xy = (0.05, 0.75), xycoords = "axes fraction", color = colours_h[0], fontsize = fs)
                        ax_rel.annotate(r'$H_2$', xy = (0.05, 0.1), xycoords = "axes fraction", color = colours_h[1], fontsize = fs)
                except NameError:
                    pass
            ax.set_xlim(left = 0)

            if figure1:
                ax.set_ylim([0,200])

            fig.subplots_adjust(right = 0.99, left = 0.06, bottom = 0.08, top = 0.94, wspace = 0.02, hspace = 0.05)
            
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylim(bottom = 0)
            ax.get_xaxis().set_ticks([])
            ax_rel.set_xlabel(r'time', fontsize = fs)
            ax_rel.set_yticks(ax_rel.get_yticks()[:-1])
            ax_rel.set_ylim([0,1])
            ax_rel.set_xlim(left = 0)
            if index_method == 0:
                ax_rel.set_ylabel(r'relative abundance', fontsize = fs)

if plot_simulation or plot_ODE or plot_SDE:
    plt.show()




