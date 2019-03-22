"""
This script can generate Fig2 and Fig3 from data stored in data_main and data_w.

This script calculates average and variance from simulated extinction times (stored in 'data' folders)
Fig2: extinction times (all methods) are comapred with approximate but analytical formula (drift method, see Supplement)
Fig3: extinction times (dt methods) are compared with sojourn times (numerically exact. see mathematica notebook) 

run with python3
"""

import numpy as np
import sys
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import csv

figure2 = True # either (plot_exttime)
figure3 = False # or (plot_exttime_w)
figure3b = False # other representation of the data. figure3=True necessary

if figure2:
  numberofruns = 1000
  folder = 'data_main'  

  plot_exttime_w = 0

  plot_exttime = 1
  plot_simulation_mean = 1
  plot_simulation_median = 0
  plot_simulation_mode = 0

  plot_shaded_stddev = 0
  plot_violin = 0
  plot_histo = 1  
  plot_drifttime = 1  
  number_of_bins = 100

  logscale = 1

if figure3:
  numberofruns = 1000
  folder = 'data_w'  

  plot_exttime_w = 1

  plot_exttime = 0
  plot_simulation_mean = 0
  plot_simulation_median = 0
  plot_simulation_mode = 0

  plot_shaded_stddev = 0
  plot_violin = 0
  plot_histo = 1  
  plot_drifttime = 0
  number_of_bins = 100

  logscale = 1

import collections
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
####################################################################################################################################


####################################################################################################################################
#### paras  CAREFUL - THIS ONLY HOLDS FOR CONSTANT birth_H but this depends on N_P which we are changing on the x-axis!
birth_H = 0.5  # 0.5 (dafault) or higher [0.5, 1.0, ] --> p* higher but not h* --> changes N_P but not N_H
death_P = 1.
carrying_capacity_LV = 500.
mu     = birth_H / carrying_capacity_LV  # competition rate is birth rate scales with carrying capacity
# reactions with two substrates are scaled 
# (second reactant number would increase rate. thus we take relative number. relative to volume? here we use carrying capacity)
lam0   = 4.  # base infection rate
lam    = lam0 / carrying_capacity_LV  # infection rate scaled with carrying capacity 

h_star = death_P / lam
p_star = birth_H / lam * (1 - 2 * (death_P / lam) / (birth_H / mu))

N_H = round(2 * h_star)  # total host population
N_P = round(2 * p_star)
w_H = 0.5  # selection intensity host in [0,1]
w_P = 1 
alpha = 1  # impact of matching type (p1 (p2) infects h1 (h2))
beta  = 0  # impact of cross-infection (p1 (p2) infects h2 (h1))
Mh = np.array([[beta, alpha], [alpha, beta]])  # impact of parasite on host Mh = - Mp ^ T + alpha + beta
Mp = np.array([[alpha, beta], [beta, alpha]])  # impact of host on parasite

####################################################################################################################################

#### main loops through this
methods = ['moran', 'local_update', 'gillespie_moran', 'gillespie_local_update', 'song_JTB', 'logistic_Lotka_Volterra', 'Lotka_Volterra']
method_names = [r'discrete time'+'\n'+r'Moran', r'discrete time'+'\n'+r'Pairwise'+'\n'+r'Comparison', r'Moran',  r'Pairwise'+'\n'+r'Comparison', r'Self controlling', r'Logistic'+'\n'+r'Independent'+'\n'+r'Interactions', r'Independent'+'\n'+r'Interactions']

N_Ps = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])

####
total_methods = len(methods)
total_N_Ps = len(N_Ps)
matrix_mmv = np.zeros((4,total_methods,total_N_Ps))  # mean, median, std deviation
runs = np.zeros((total_methods,total_N_Ps))
runs.astype(int)

####################################################################################################################################
#### plot options
lines = ["-","--",":","-."]  # lines for t_drift, sojourn_edge, sojourn_corner
lw = 2  # linewidth
markers = ["o",",","*","v","^","o"]

colors = ['#006699','#8bb92d', '#5bc9ff','#0d9648','#480d96','#ff748c','#960D5B'] # b lg lb g
####################################################################################################################################
if plot_exttime_w == 1:
  method_linestyle = ['-', '--']
  method_linestyle2 = ['-', '--']
  method_marker = ['s','x']
  sizes = [15,30]
  methods = ['moran', 'local_update']
  wps = [0.2, 0.4, 0.6, 0.8, 1.0]
  whs = [i for i in wps]
  Ns = [50, 100, 150]
  colors = cm.viridis(np.linspace(0, 1, len(whs)))
  fig, all_ax = plt.subplots(ncols = len(Ns), nrows = 1, figsize = [12,3.3])
  matrix_exttime_w = np.zeros((len(whs), len(wps), len(methods), len(Ns), 4))
  matrix_exttime_w.fill(np.nan)
  all_ax[0].annotate(r'Paiwise comparison', xy = (0.05, 0.75), xycoords = "axes fraction")
  all_ax[0].annotate(r'Moran', xy = (0.7, 0.075), xycoords = "axes fraction")

  if figure3b:
    lattice = np.zeros((len(wps), len(whs)))
    cmap = plt.cm.get_cmap('viridis')
    fig_b, ax_b = plt.subplots(nrows = 2, ncols = 3)

  for index_N, N in enumerate(Ns):
    try:
      ax = all_ax[index_N]
      if logscale == 1:
        ax.set_yscale('log')
    except TypeError:
      ax = all_ax
    ax.annotate(r'$N_H=N_P={}$'.format(N), xy = (0.02, 0.93), xycoords = "axes fraction")
    N_H = N
    N_P = N
    divideby = N#**2
    ax.scatter([],[],marker = '.', color = 'none', label = r'$w_H$')
    for index_wH, w_H in enumerate(whs):
      ax.scatter([],[],marker = '.', color = colors[index_wH], label = str(w_H))

    for index_method, method in enumerate(methods):
      for index_wH, w_H in enumerate(whs):
        for index_wP, w_P in enumerate(wps):
          file_name = './'+folder+'/'+method+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_wH{:.3f}'.format(w_H)+'_wP{:.3f}'.format(w_P)+'_alpha{:d}'.format(alpha)+'_beta{:d}'.format(beta)+'.txt'
          try:
            file_times = open(file_name)
          except FileNotFoundError:
            break
          contents = np.array([float(line.rstrip()) for line in file_times])
          file_times.close()
          matrix_exttime_w[index_wH, index_wP, index_method, index_N, 0] = np.mean(contents)
          matrix_exttime_w[index_wH, index_wP, index_method, index_N, 1] = math.sqrt(np.var(contents))
          matrix_exttime_w[index_wH, index_wP, index_method, index_N, 2] = len(contents)
        
        ax.scatter([index_wH/4*0.1-0.05+0.0125*index_method+wp for wp in wps], matrix_exttime_w[index_wH, :, index_method, index_N, 0]/divideby, marker = method_marker[index_method], s = sizes[index_method], color = colors[index_wH], zorder = 2)
        errorbar_localupdate = ax.errorbar([index_wH/4*0.1+0.0125*index_method-0.05+wp for wp in wps], matrix_exttime_w[index_wH, :, index_method, index_N, 0]/divideby, yerr=matrix_exttime_w[index_wH, :, index_method, index_N, 1]/divideby, fmt = 'none', elinewidth = 0.5, capsize = 1.5, ecolor = colors[index_wH])
        errorbar_localupdate[-1][0].set_linestyle(method_linestyle[index_method])
      ### Fig b
      if figure3b:
        lattice[:,:] = matrix_exttime_w[:,:,index_method, index_N, 0]
        image = ax_b[index_method, index_N].imshow(lattice, cmap = cmap, interpolation = 'nearest', origin = 'lower')
        ax_b[index_method, index_N].set_xlabel(r'$w_P$')
        ax_b[index_method, index_N].set_ylabel(r'$w_H$')
        plt.colorbar(image, ax = ax_b[index_method, index_N])
        ax_b[index_method, 0].annotate(method_names[index_method], xy = (-1, 0.5), xycoords = "axes fraction")
        ax_b[0, index_N].annotate('N='+str(N), xy = (0.1, 1.05), xycoords = "axes fraction")
      ### SOJOURN
      with open('./sojourn_'+method+'_N{:d}'.format(N)+'.txt', "r") as f:
        csvraw = list(csv.reader(f))

      whs_sojourn = np.array([row[0] for row in csvraw[1:]]).astype(float)
      wps_sojourn = np.array([row[1] for row in csvraw[1:]]).astype(float)
      sojourn_edge = np.array([row[2] for row in csvraw[1:]]).astype(float)
      for index_sojourn_file, (wh_sojourn, wp_sojourn) in enumerate(zip(whs_sojourn, wps_sojourn)):
        if wh_sojourn in whs and wp_sojourn in wps:
          index_wH = [i for i, val in enumerate(whs) if val == wh_sojourn][0]
          index_wP = [i for i, val in enumerate(wps) if val == wp_sojourn][0]
          matrix_exttime_w[index_wH, index_wP, index_method, index_N, 3] = sojourn_edge[index_sojourn_file]
      for index_wH, wH in enumerate(whs):  
        ax.plot([index_wH/4*0.1-0.05+0.0125*index_method+wp for wp in wps], matrix_exttime_w[index_wH, :, index_method, index_N, 3]/divideby, linestyle = method_linestyle2[index_method], color = colors[index_wH], zorder = 0, linewidth = 0.8)
  
    # end method loop but still in N-loop (for different subplots)
    print('N', N, ' runs ', matrix_exttime_w[:,:,:,index_N,2])
    # ax.set_ylabel(r'ext time')
    ax.set_xlim([wps[0]-0.06, wps[-1]+0.06])
    ax.set_xlabel(r'$w_P$', fontsize = 12)
    min_ext = min(matrix_exttime_w[:,:,:,index_N,0].flatten())
    max_ext = max(matrix_exttime_w[:,:,:,index_N,0].flatten())
    diff_ext = max_ext - min_ext
    # ax.set_ylim([10**(1),10**3])
  # all_ax[0].set_ylim(bottom = 3*10**2, top = 10**4)
  all_ax[0].set_ylim(bottom = 5, top = 3*10**2)
  # all_ax[1].set_ylim(top = 10**6)
  all_ax[1].set_ylim(bottom = 5, top = 10**4)
  # all_ax[2].set_ylim(bottom = 10**3, top = 10**7)
  all_ax[2].set_ylim(bottom = 5, top = 10**4)
  # end all loops
  my_legend = ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))
  try:
    all_ax[0].set_ylabel(r'extinction time', fontsize = 12)
    fig.subplots_adjust(left = 0.07, bottom = 0.13, right = 0.87, top = 0.98, wspace = 0.23)

  except TypeError:
    ax.set_ylabel(r'ext time')
    if logscale == 1:
      ax.set_yscale('log')
  if logscale == 0:
    all_ax[0].set_ylim([0, 5000])
    all_ax[1].set_ylim([0,15000])
    all_ax[2].set_ylim([0,40000])
  plt.setp(all_ax, xticks=[0.2, 0.4, 0.6, 0.8, 1.0], xticklabels=['0.2', '0.4', '0.6', '0.8', '1'])

####################################################################################################################################
#### loop through simulation results
if plot_exttime == 1:
  fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize= (15.5, 7))

  #### for all 6 methods
  for index_method, method in enumerate(methods):
    
    if method == 'Lotka_Volterra':
      mu0 = 0.00
      method = 'logistic_Lotka_Volterra'
    else:
      mu0 = 1
    for index_N_P, N_P in zip(np.arange(total_N_Ps), N_Ps):
      ####################################################################################################################################
      #### open files
        if method == 'logistic_Lotka_Volterra':
          if mu0 > 0:  # real logistic_LV
            birth_H = (carrying_capacity_LV * lam ** 2 * N_P) / (2 * (carrying_capacity_LV * lam - 2 * death_P))
            mu  = birth_H / carrying_capacity_LV  
          else:  # LV without logistic growth ( mu = 0 )
            mu = 0
            birth_H = N_P * lam / 2
          file_name = method+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_mu{:.3f}'.format(mu)+'_lam{:.3f}'.format(lam)+'_lam0{:.3f}'.format(lam0)+'_birthH{:.3f}'.format(birth_H)+'_deathP{:.3f}'.format(death_P)+'_carryingCapLV{:.3f}'.format(carrying_capacity_LV)+'.txt'
          try:
            file_times = open('./'+folder+'/'+file_name)
          except FileNotFoundError:
            print('no file')
            print('file name', file_name)
            continue  # continue with next N_P in for-loop
        else:
          file_name = method+'_NH{:d}'.format(N_H)+'NP{:d}'.format(N_P)+'_wH{:.3f}'.format(w_H)+'_wP{:.3f}'.format(w_P)+'_alpha{:d}'.format(alpha)+'_beta{:d}'.format(beta)+'.txt'
          try:
            file_times = open('./'+folder+'/'+file_name)
          except FileNotFoundError:
            print('no file')
            print('file name', file_name)
            continue  # continue with next N_P in for-loop
        contents = np.array([float(line.rstrip()) for line in file_times])
        contents = contents[:numberofruns]
        if method in ['moran', 'local_update']:
          fracNP = N_P ** 2 / (N_P * N_H)
          fracNH = (N_H - N_P) * N_P / (N_H * N_P)
          contents = contents #/ (N_P / fracNP + N_H / fracNH) #/((N_H * N_P)) #/ N_H  #/ (N_H * N_P)  #((N_H + N_P) / 2)
          
        ####################################################################################################################################
        #### plot histogram or violins
        if plot_histo == 1:
          current_mean = np.mean(contents)
          if logscale == 1:
            hist, bin_data = np.histogram(contents, bins = 10 ** np.linspace(np.log10(current_mean*0.1), np.log10(current_mean*10), number_of_bins)) 
          else:
            hist, bin_data = np.histogram(contents, bins = np.linspace(current_mean*0.1, current_mean*10, number_of_bins)) 
          y_values = (bin_data[0:-1] + bin_data[1:]) / 2
          y_values_all = flatten([[yvalue] * numberofpoints for yvalue, numberofpoints in zip(y_values, hist) ])
          x_values_all = flatten([[N_P + (i-(j-1)/2)/5 for i in range(j)] for j in hist])
          x_values_outer = flatten([[N_P + (i-(j-1)/2)/5 for i in [0, j-1] ] for j in hist])
          x_values_left = flatten([[N_P + (0-(j-1)/2)/5 ] for j in hist])
          x_values_right = flatten([[N_P + ((j-1)/2)/5 ] for j in hist])
          ax.plot(x_values_left,  y_values, color = colors[index_method], alpha = 0.3)
          ax.plot(x_values_right, y_values, color = colors[index_method], alpha = 0.3)
          ax.fill_betweenx(y_values, x_values_left, x_values_right, color = colors[index_method], alpha = 0.03)

        if plot_violin == 1:
          vplot = ax.violinplot(contents, positions = [N_P], vert = True, widths = 3, showmeans = False, showextrema = False, showmedians = False) 
          for patch in vplot['bodies']:
            patch.set_color(colors[index_method])


        ####################################################################################################################################
        #### calculate result
        matrix_mmv[0, index_method, index_N_P] = np.mean(contents)
        matrix_mmv[1, index_method, index_N_P] = np.median(contents)
        matrix_mmv[2, index_method, index_N_P] = math.sqrt(np.var(contents)) #std dev
        histo = np.histogram(contents, bins = 500)
        matrix_mmv[3, index_method, index_N_P] = histo[1][np.argmax(histo[0])]
        runs[index_method, index_N_P] = len(contents)
        
        ######## annotate
        last = N_Ps[-1]
        # method_names = [r'discrete time'+'\n'+r'Moran', r'discrete time'+'\n'+r'Pairwise'+'\n'+r'Comparison', r'Moran',  r'Pairwise'+'\n'+r'Comparison', r'Self sustainig', r'Logistic'+'\n'+r'Independent'+'\n'+r'Interactions', r'Independent'+'\n'+r'Interactions']
        ax.annotate("dtPC",     xy=(105,1.2*10**4), color = colors[1])
        ax.annotate("dtMoran",  xy=(105,8.3*10**3), color = colors[0])

        ax.annotate("Moran",    xy=(100.4,3*10**2), color = colors[2])

        ax.annotate("LogIR",    xy=(last+3,7*10**2), color = colors[5])
        ax.annotate("PC",       xy=(last+6,1.5*10**2), color = colors[3])
        ax.annotate("SCPS",     xy=(last+4,9.9*10**1), color = colors[4])
        ax.annotate("IR",       xy=(last+6,6*10**1), color = colors[6])


  runs = runs.astype(int)
  print("runs : ", runs)

####################################################################################################################################
#### drift time approximated with average change of constant of motion dH. for discrete time processes moran and local update
if plot_drifttime == 1:
  drift_N_Ps = np.array([30, 40, 50, 60, 70, 80, 90, 100])
  drift_total_N_Ps = len(drift_N_Ps)

  def dHmoran(w_H, w_P, N_H, N_P):
    A = - 200 * (N_H ** 2 + N_P ** 2 - 2)
    B = 4 - 2 * N_P ** 2 + N_H      * (2 * N_P ** 2 - 4)
    C = 4 - 4 * N_P      + N_H      * (4 * N_P      - 4)
    D = 4 - 4 * N_P      + N_H ** 2 * (2 * N_P      - 2)
    return (A + B * w_H ** 2  - C * w_H * w_P + D * w_P ** 2) / (225 * N_H ** 2 * N_P ** 2)

  def dHLU(w_H, w_P, N_H, N_P):
    return (- 100 * (N_H ** 2 + N_P ** 2 - 1) - 4 * N_H * N_P * w_H * w_P) / (225 * N_H ** 2 * N_P ** 2)
  

  if 'moran' in methods:
    collected_drifttimes = np.zeros((drift_total_N_Ps, 1))
    index = [index for index, method in enumerate(methods) if method == 'moran'][0]
    for index_N_P, N_P in zip(np.arange(drift_total_N_Ps), drift_N_Ps):
      drifttime = - 1 / dHmoran(w_H, w_P, N_H, N_P)
      print(drifttime)
      collected_drifttimes[index_N_P] = drifttime
    collected_drifttimes[collected_drifttimes<0]='nan'
    ax.plot(drift_N_Ps, collected_drifttimes, '-', color = colors[index], markersize = 10)

  if 'local_update' in methods:
    collected_drifttimes = np.zeros((drift_total_N_Ps, 1))
    index = [index for index, method in enumerate(methods) if method == 'local_update'][0]
    for index_N_P, N_P in zip(np.arange(drift_total_N_Ps), drift_N_Ps):
      drifttime = - 1 / dHLU(w_H, w_P, N_H, N_P)
      collected_drifttimes[index_N_P] = drifttime
    collected_drifttimes[collected_drifttimes<0]='nan'
    ax.plot(drift_N_Ps, collected_drifttimes, '-', color = colors[index], markersize = 10)

####################################################################################################################################
#### std dev
if plot_shaded_stddev == 1:
  for index_method, method in enumerate(methods):
    ax.plot(N_Ps, matrix_mmv[0, index_method, :], 'o-', color =colors[index_method])
    ax.fill_between(N_Ps, matrix_mmv[0, index_method, :] - matrix_mmv[2, index_method, :], matrix_mmv[0, index_method, :] + matrix_mmv[2, index_method, :], alpha = 0.3, color = colors[index_method])
####################################################################################################################################
#### plot 

if plot_exttime == 1:
  for index_method, method in enumerate(methods):  # mean, median
    if plot_simulation_mean == 1: # and matrix_mmv[0,index_method,index_N_P]>1:
      ax.scatter(N_Ps,matrix_mmv[0,index_method,:],color = colors[index_method],marker = markers[0], s = 30, label = method_names[index_method], zorder = 900)  # mean
    
    if plot_simulation_median == 1:
      ax.scatter(N_Ps,matrix_mmv[1,index_method,:],color = colors[index_method],marker = markers[1], s = 20)  # median
    
    if plot_simulation_mode == 1:
      ax.scatter(N_Ps,matrix_mmv[3,index_method,:],color = colors[index_method],marker = markers[2], s = 20)  # mode
  

  ## axes
  ax.set_xlabel(r"$N_P$", fontsize = 15)
  ax.set_ylabel(r"extinction time", fontsize=15)
  if logscale == 1:

      ax.set_yscale('log')
      ax.set_ylim([10**0,2*10**5]) 

plt.show()