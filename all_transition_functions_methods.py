import numpy as np
import bisect

##########################################################################################################################################################
# method:

# dtPC    (discrete time) 
# dtMoran (discrete time)        
# PC       
# Moran            

# SCPS  (4D)                
# logIR (4D) 
# IR    (4D) 

##########################################################################################################################################################
def transitions(current_method, N_H, N_P, Mh, Mp, w_H, w_P):

    if current_method in ['dtPC', 'dtMoran', 'PC', 'Moran']:

        # integer abundances
        js_H    = np.arange(0, N_H + 1, 1)  # j = 0 to N_H (H1)
        njs_H   = js_H[::-1]  # j = N_H to 0 (H2)
        jsnjs_H = np.array([js_H, njs_H]).T  # j = 0 to N_H and N_H to 0 (H1 and H2)
        js_P    = np.arange(0, N_P + 1, 1)  # j = 0 to N_P
        njs_P   = js_P[::-1]  # j = N_P to 0
        jsnjs_P = np.array([js_P, njs_P]).T  # j = 0 to N_P and N_P to 0

        # payoffs pi_H = M_H * p and pi_P = M_P * h
        piH_plus  = np.array([Mh[0, :].dot(i) for i in jsnjs_P]) * 1. / N_P  # vector for every p 
        piH_minus = np.array([Mh[1, :].dot(i) for i in jsnjs_P]) * 1. / N_P
        piP_plus  = np.array([Mp[0, :].dot(i) for i in jsnjs_H]) * 1. / N_H  # for every h
        piP_minus = np.array([Mp[1, :].dot(i) for i in jsnjs_H]) * 1. / N_H

        # fitness is weighted function of payoff with selection intensity w_H and w_P
        fH_plus  = 1 - w_H + w_H * piH_plus   # for every p
        fH_minus = 1 - w_H + w_H * piH_minus
        fP_plus  = 1 - w_P + w_P * piP_plus 
        fP_minus = 1 - w_P + w_P * piP_minus 
        # average fitness fH for host and fP parasite
        fH_avg =  np.outer(js_H * 1. / N_H, fH_plus) + np.outer(njs_H * 1. / N_H, fH_minus)  # for every h (rows) and p (columns)
        fP_avg = (np.outer(js_P * 1. / N_P, fP_plus) + np.outer(njs_P * 1. / N_P, fP_minus)).transpose()

        # make correct shape (host = rows, parasite = columns)
        fH_plus  = np.repeat(fH_plus[np.newaxis, :],  N_H + 1, 0)  # shape N_P -> add shape N_H
        fH_minus = np.repeat(fH_minus[np.newaxis, :], N_H + 1, 0)
        fP_plus  = np.repeat(fP_plus[:, np.newaxis],  N_P + 1, 1)  # shape N_H -> add shape N_P
        fP_minus = np.repeat(fP_minus[:, np.newaxis], N_P + 1, 1)
        jsmatrixh  = np.repeat(js_H[:, np.newaxis],  N_P + 1, 1)  # repeat vector js_H for all N_P columns
        jsmatrixp  = np.repeat(js_P[np.newaxis, :],  N_H + 1, 0)
        njsmatrixh = np.repeat(njs_H[:, np.newaxis], N_P + 1, 1)
        njsmatrixp = np.repeat(njs_P[np.newaxis, :], N_H + 1, 0)
        # now everything has shape (N_H, N_P) - in that order because of where the newaxis was added

        ##### MORAN PROCESS ###
        # divide by average fitness
        if current_method in ['dtMoran', 'Moran']:
            fH_avg[0, 0] = 10 ** (-5)
            fH_avg[N_H, N_P] = 10 ** (-5)
            fP_avg[0, N_P] = 10 ** (-5)
            fP_avg[N_H, 0] = 10 ** (-5)
            phiH_plus  = fH_plus / fH_avg # matrix for every h and p
            phiH_minus = fH_minus / fH_avg
            phiP_plus  = fP_plus / fP_avg
            phiP_minus = fP_minus / fP_avg
            # print('phis')
            # print(phiH_plus)
            # print(phiH_minus)
            # print(phiP_plus)
            # print(phiP_minus)

        ##### LOCAL UPDATE ###
        # average fitness has no influence only difference (local process)
        elif current_method in ['dtPC', 'PC']:
            delta_piH = piH_plus - piH_minus
            delta_piP = piP_plus - piP_minus
            phiH_plus  = np.full((N_H + 1, N_P + 1), 0.5)  + 0.5 * (fH_plus - fH_minus) / max(delta_piH)
            phiH_minus = np.full((N_H + 1, N_P + 1), 0.5)  + 0.5 * (fH_minus - fH_plus) / max(delta_piH)
            phiP_plus  = np.full((N_H + 1, N_P + 1), 0.5)  + 0.5 * (fP_plus - fP_minus) / max(delta_piP)
            phiP_minus = np.full((N_H + 1, N_P + 1), 0.5)  + 0.5 * (fP_minus - fP_plus) / max(delta_piP)
            # print('phis')
            # print(phiH_plus)
            # print(phiH_minus)
            # print(phiP_plus)
            # print(phiP_minus)
            
        ##### DISCRETE TIME ###
        # use relative numbers js/N in [0,1] and njs/N in [0,1]
        if current_method in ['dtPC', 'dtMoran']:  
            Th_plus  = phiH_plus  * jsmatrixh * 1. / N_H * njsmatrixh * 1. / N_H  # for every h (rows) and p (columns)
            Th_minus = phiH_minus * jsmatrixh * 1. / N_H * njsmatrixh * 1. / N_H  # for every h (rows) and p (columns)
            Th_0     = 1. - Th_plus - Th_minus
            Tp_plus  = phiP_plus  * jsmatrixp * 1. / N_P * njsmatrixp * 1. / N_P  # for every h (rows) and p (columns) 
            Tp_minus = phiP_minus * jsmatrixp * 1. / N_P * njsmatrixp * 1. / N_P # for every h (rows) and p (columns)
            Tp_0     = 1. - Tp_plus - Tp_minus
            return np.array([Th_0 * Tp_0, Th_plus * Tp_0, Th_0 * Tp_plus, Th_minus * Tp_0, Th_0 * Tp_minus, Th_plus * Tp_plus, 
                Th_minus * Tp_minus, Th_plus * Tp_minus, Th_minus * Tp_plus])

        ##### GILLESPIE ###
        # gillespie algorithm: reaction with two substrates js and njs (integers/actual number of 'molecules') 
        # divide by 'volume' here N otherwise that reaction becomes more probable than reactions with only single reactant
        elif current_method in ['PC', 'Moran']:  
            Th_plus  = phiH_plus  * jsmatrixh * 1. / N_H * njsmatrixh 
            Th_minus = phiH_minus * jsmatrixh * 1. / N_H * njsmatrixh
            Tp_plus  = phiP_plus  * jsmatrixp * 1. / N_P * njsmatrixp
            Tp_minus = phiP_minus * jsmatrixp * 1. / N_P * njsmatrixp
            return np.array([Th_plus, Th_minus, Tp_plus, Tp_minus])

    else:  # 4D methods song_JTB or logistic_Lotka_Volterra. here we cannot precompute transitions
        return 'nan'
##########################################################################################################################################################
def current_transitions_4d_scps(n_h, n_h2, n_p, n_p2, w_H, w_P, alpha, beta):

    # in song death_host and birth_parasite depend on the interactions defined by the payoff matrix (alpha, beta \\ beta, alpha). 
    # we also weigh the payoff with the selection intensity w_h, w_P
    # birth_host is avergae of death_host and death_parasite is the average of birth_parasite to keep population approximately constant
    
    death_H1 = (1 - w_H + w_H * (n_p * alpha + n_p2 * beta)  / (n_p + n_p2))
    # print("dH1", death_H1, w_H, n_p, n_p2)
    death_H2 = (1 - w_H + w_H * (n_p * beta  + n_p2 * alpha) / (n_p + n_p2))
    birth_H  = (n_h * death_H1 + n_h2 * death_H2) / (n_h + n_h2)  # average birth is defined as average death
    birth_P1 = (1 - w_P + w_P * (n_h * alpha + n_h2 * beta)  / (n_h + n_h2))
    birth_P2 = (1 - w_P + w_P * (n_h * beta  + n_h2 * alpha) / (n_h + n_h2))
    death_P  = (n_p * birth_P1 + n_p2 * birth_P2) / (n_p + n_p2)  # average death is defined as average birth
    
    # now transition probabilities are the rates multiplied with the abundances
    Th1_plus  = n_h  * birth_H
    Th2_plus  = n_h2 * birth_H
    Th1_minus = n_h  * death_H1
    Th2_minus = n_h2 * death_H2    
    Tp1_plus  = n_p  * birth_P1
    Tp2_plus  = n_p2 * birth_P2
    Tp1_minus = n_p  * death_P
    Tp2_minus = n_p2 * death_P
    return np.array([Th1_plus, Th2_plus, Th1_minus, Th2_minus, Tp1_plus, Tp2_plus, Tp1_minus, Tp2_minus])

def current_transitions_4d_IR(n_h, n_h2, n_p, n_p2, birth_H_LV, death_P_LV, mu, lam):

    # strict chemical reactions happen with fixed rates
    Th1_plus  = n_h * birth_H_LV
    Th2_plus  = n_h2 * birth_H_LV
    Th1_minus = n_h * (mu * (n_h + n_h2) + lam * n_p)
    Th2_minus = n_h2 * (mu * (n_h + n_h2) + lam * n_p2)
    Tp1_plus  = n_p  * (lam * n_h)
    Tp2_plus  = n_p2 * (lam * n_h2)
    Tp1_minus = n_p  * death_P_LV
    Tp2_minus = n_p2 * death_P_LV

    return np.array([Th1_plus, Th2_plus, Th1_minus, Th2_minus, Tp1_plus, Tp2_plus, Tp1_minus, Tp2_minus])

##########################################################################################################################################################
def update(current_method, current_probabilities):

    if current_method in ['dtMoran', 'dtPC']:
        delta_h_values = np.array([0, 1, 0, -1,  0, 1, -1,  1, -1])
        delta_p_values = np.array([0, 0, 1,  0, -1, 1, -1, -1,  1])

        cumulative_probabilities = np.cumsum(current_probabilities)
        r = np.random.rand(1)
        index = bisect.bisect_right(cumulative_probabilities,r)  # find position upper bound of random number in cumulative probabilies
        delta_t = 1

    elif current_method in ['Moran', 'PC']:
        delta_h_values = np.array([1, -1, 0,  0])
        delta_p_values = np.array([0,  0, 1, -1])
        # print(current_probabilities)
        times = [1 / i * np.log(1 / np.random.rand(1)) if i!=0 else 'nan' for i in current_probabilities]
        # print(times)
        index = np.nanargmin(times) # find position of first event
        delta_t = times[index]
    delta_h = delta_h_values[index]  
    delta_p = delta_p_values[index]
    # print("dh", delta_h)
    return delta_h, delta_p, delta_t

##########################################################################################################################################################
def update_4d(current_probabilities):
    delta_h_values  = np.array([1, 0, -1,  0, 0, 0,  0,  0])
    delta_p_values  = np.array([0, 0,  0,  0, 1, 0, -1,  0])
    delta_h2_values = np.array([0, 1,  0, -1, 0, 0,  0,  0])
    delta_p2_values = np.array([0, 0,  0,  0, 0, 1,  0, -1])
    # print("probs", current_probabilities)
    times = [1 / i * np.log(1 / np.random.rand(1)) if i!=0 else 900000 for i in current_probabilities]
    index = np.nanargmin(times) # find position of first event
    # print("i", index)

    delta_h  = delta_h_values[index]  
    delta_p  = delta_p_values[index]
    delta_h2 = delta_h2_values[index]  
    delta_p2 = delta_p2_values[index]
    delta_t  = times[index]
    # print(delta_t)
    return delta_h, delta_h2, delta_p, delta_p2, delta_t

