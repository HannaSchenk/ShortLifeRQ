import numpy as np
import bisect

##########################################################################################################################################################

def transitions(n_h, n_p, N_H, N_P, Mh, Mp, w_H, w_P, mutation_rate_host, mutation_rate_parasite):

    alpha = Mp[0, 0]
    beta = Mp[1, 0]
    num_types = len(Mp)
    number_of_parasites = sum([i>0 for i in n_p])
    number_of_hosts = sum([j>0 for j in n_h])

    # scaling
    const = (alpha + (number_of_hosts-1) * beta)/(beta + (number_of_parasites-1) * alpha)

    Mh = Mh * const


    pi_H = np.dot(Mh, n_p) / N_P  # vector
    pi_P = np.dot(Mp, n_h) / N_H  # vector
    fH = 1 - w_H + w_H * pi_H  # vector
    fP = 1 - w_P + w_P * pi_P  # vector
    fH_avg = np.inner(n_h / N_H, fH)  # scalar
    fP_avg = np.inner(n_p / N_P, fP)  # scalar
    # fH_avg = sum([n_h_pos * fH_pos if n_h_pos>0 for n_h_pos, fH_pos in zip(n_h, fH)])
    if fH_avg == 0:  # then also birthH=0
        fH_avg = 10**(-5)
    if fP_avg == 0:
        fP_avg = 10**(-5)
    phi_H = fH / fH_avg  # vector
    phi_P = fP / fP_avg  # vector
    # choose for birth:
    birth_H = n_h / N_H * phi_H  # vector (element wise multiplication)
    birth_P = n_p / N_P * phi_P
    # choose for death:
    death_H = n_h / N_H  # vector ... relative abundance \in [0,1]
    death_P = n_p / N_P

    Th_plus_minus = np.outer(birth_H, death_H) * N_H # matrix
    Tp_plus_minus = np.outer(birth_P, death_P) * N_P

    Th_mutations_plus = n_h * mutation_rate_host/2
    Th_mutations_minus = n_h * mutation_rate_host/2
    Tp_mutations_plus = n_p * mutation_rate_parasite/2
    Tp_mutations_minus = n_p * mutation_rate_parasite/2
    T_mutation = np.array([Th_mutations_plus, Th_mutations_minus, Tp_mutations_plus, Tp_mutations_minus]).flatten()
    return Th_plus_minus, Tp_plus_minus, T_mutation

##########################################################################################################################################################
def update_gillespie_moran(Th_plus_minus, Tp_plus_minus, T_mutation, num_types, mutate):
    if mutate == 1:
        # proplist2 = np.array([Th_plus_minus, Tp_plus_minus]).flatten()
        # proplist3 = T_mutation.flatten()
        proplist4 = [list(np.array([Th_plus_minus, Tp_plus_minus]).flatten()), list(T_mutation.flatten())]
        proplist1 = [i for sublist in proplist4 for i in sublist]
    else:
        proplist1 = list(np.array([Th_plus_minus, Tp_plus_minus]).flatten())
    if any([i<0 for i in proplist1]):
        print('rates', proplist1)
    
    ##########################################################################################################################################################
    times = [1 / i * np.log(1 / np.random.rand(1)) if i!=0 else 900000 for i in proplist1]
    if all([i==900000 for i in times]):
        print('all probabilites are zero')
        quit()
    index = np.nanargmin(times) # find position of first event# print("dh", delta_h)
    delta_t  = times[index]
    if delta_t<0:
        print('time<0')
        quit()
    delta_h = np.zeros(num_types)
    delta_p = np.zeros(num_types)
    # print('i', index, 'dt', delta_t)
    # something happened in host population
    if index < num_types ** 2:  
        # print('no')                        
        index_death_H = index % num_types
        index_birth_H = int((index - index_death_H) / num_types)
        delta_h[index_death_H] += -1
        delta_h[index_birth_H] += 1
        # print(index, index_death_H, index_birth_H, delta_h)

    # something in parasite population
    elif index < (num_types ** 2) * 2:       
        # print('no')                        

        index = index - num_types ** 2                                    
        index_death_P = index % num_types
        index_birth_P = int((index - index_death_P) / num_types)
        delta_p[index_death_P] += -1
        delta_p[index_birth_P] += 1
    
    # mutation
    else:
        # print('mutation index', index)
        index = index - (num_types ** 2) * 2
        # print('converted index', index)
        # host mutation i --> i+1 forward
        if index < num_types-1:
            delta_h[index]       += -1
            delta_h[index+1]     +=  1
            print('h', index, index+1)
        elif index == num_types - 1:
            delta_h[-1]          += -1
            delta_h[0]           +=  1
            print('h', num_types-1, 0) 
        # host mutation i --> i-1 backward
        elif index == num_types: 
            delta_h[0]               += -1
            delta_h[-1]              +=  1
            print('h', 0, num_types-1)
        elif index < num_types * 2:
            delta_h[index-num_types]   += -1
            delta_h[index-num_types-1] +=  1
            print('h', index-num_types, index-num_types-1)

        # parasite mutation i --> i+1 forward
        elif index < num_types * 3 - 1: 
            delta_p[index-num_types*2]   += -1
            delta_p[index+1-num_types*2] +=  1
            print('p', index-num_types*2, index+1-num_types*2)
        elif index == num_types * 3 - 1:
            delta_p[-1]                  += -1
            delta_p[0]                   +=  1
            print('p', num_types-1, 0)

        # parasite mutation i --> i-1 backward
        elif index == num_types * 3:
            delta_p[0]       += -1
            delta_p[-1]      +=  1
            print('p', 0, num_types-1)
        else:
            delta_p[index-num_types*3]   += -1
            delta_p[index-num_types*3-1] +=  1
            print('p', index-num_types*3, index-num_types*3-1)    




    # print("dh",delta_h, delta_p)
    return delta_h, delta_p, delta_t
##########################################################################################################################################################
def current_transitions_4d_song(n_h, n_p, w_H, w_P, Mh, Mp, num_types, mutation_rate_host, mutation_rate_parasite):



    # in song death_host and birth_parasite depend on the interactions defined by the payoff matrix (alpha, beta \\ beta, alpha). 
    # we also weigh the payoff with the selection intensity w_h, w_P
    # birth_host is avergae of death_host and death_parasite is the average of birth_parasite to keep population approximately constant
    sum_h = np.sum(n_h)  # scalar
    sum_p = np.sum(n_p)
    death_H = (1 - w_H + w_H * np.dot(Mp, n_p))  / sum_p  # vector
    birth_P = (1 - w_P + w_P * np.dot(Mp, n_h))  / sum_h
    # print("dH1", death_H[0], w_H, Mp, n_p)

    # if avg birthH or avg deathP is 0 then no
    birth_H = np.inner(death_H, n_h) / sum_h  # average birth is defined as average death  scalar
    death_P = np.inner(birth_P, n_p) / sum_p  # average death is defined as average birth
    
    # now transition probabilities are the rates multiplied with the abundances
    Th_plus  = n_h  * birth_H
    Th_minus = n_h  * death_H

    Tp_plus  = n_p  * birth_P
    Tp_minus = n_p  * death_P

    probabilites = list(np.array([Th_plus, Th_minus, Tp_plus, Tp_minus]).flatten())
    if all([i ==0 for i in probabilites]):
        if np.inner(n_h, n_p) == 0:
            print("no more matching types")
    
    Th_mutations_plus = n_h * mutation_rate_host/2
    Th_mutations_minus = n_h * mutation_rate_host/2
    Tp_mutations_plus = n_p * mutation_rate_parasite/2
    Tp_mutations_minus = n_p * mutation_rate_parasite/2
    T_mutation = np.array([Th_mutations_plus, Th_mutations_minus, Tp_mutations_plus, Tp_mutations_minus]).flatten()

    return Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation

def current_transitions_4d_LV(n_h, n_p, birth_H_LV, death_P_LV, mu, lam, mutation_rate_host, mutation_rate_parasite):

    # strict chemical reactions happen with fixed rates
    Th_plus  = n_h * birth_H_LV
    Th_minus = n_h * (mu * sum(n_h) + lam * n_p)
    Tp_plus  = n_p  * (lam * n_h)
    Tp_minus = n_p  * death_P_LV
    
    Th_mutations_plus = n_h * mutation_rate_host/2
    Th_mutations_minus = n_h * mutation_rate_host/2
    Tp_mutations_plus = n_p * mutation_rate_parasite/2
    Tp_mutations_minus = n_p * mutation_rate_parasite/2
    T_mutation = np.array([Th_mutations_plus, Th_mutations_minus, Tp_mutations_plus, Tp_mutations_minus]).flatten()

    return Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation

##########################################################################################################################################################
def update_4d(Th_plus, Th_minus, Tp_plus, Tp_minus, T_mutation, num_types, mutate):

    if mutate == 1:
        proplist = [list(np.array([Th_plus, Th_minus, Tp_plus, Tp_minus]).flatten()), list(T_mutation.flatten())]
        probabilites = [i for sublist in proplist for i in sublist]
    else:
        probabilites = list(np.array([Th_plus, Th_minus, Tp_plus, Tp_minus]).flatten())
    if any([i<0 for i in probabilites]):
        print('rates', probabilites)

    times = [1 / i * np.log(1 / np.random.rand(1)) if (i!=0 and i!='nan') else 99999999999 for i in probabilites]
    index = np.nanargmin(times) # find position of first event

    if times[index] < 0 or all([i==99999999999 for i in times]):
        return 'too much'
    delta_t  = times[index]
    if delta_t<0:
        print('time<0')
        quit()
    delta_h = np.zeros(num_types)
    delta_p = np.zeros(num_types)
    if index < num_types :
        delta_h[index] = 1
    elif index < 2 * num_types:
        delta_h[index - num_types] = -1
    elif index < 3 * num_types:
        delta_p[index - 2 * num_types] = 1
    elif index < 4 * num_types:
        delta_p[index - 3 * num_types] = -1
    else: ## MUTATION ##
        index = index - (num_types * 4)
        # print('converted index', index)
        # host mutation i --> i+1 forward
        if index < num_types-1:
            delta_h[index]       += -1
            delta_h[index+1]     +=  1
            print('mutation from h_', index+1, ' to h_', index+2)
        elif index == num_types - 1:
            delta_h[-1]          += -1
            delta_h[0]           +=  1
            print('mutation from h_', num_types, ' to h_', 1) 
        # host mutation i --> i-1 backward
        elif index == num_types: 
            delta_h[0]               += -1
            delta_h[-1]              +=  1
            print('mutation from h_', 1, ' to h_', num_types)
        elif index < num_types * 2:
            delta_h[index-num_types]   += -1
            delta_h[index-num_types-1] +=  1
            print('mutation from h_', index-num_types+1, ' to h_', index-num_types)

        # parasite mutation i --> i+1 forward
        elif index < num_types * 3 - 1: 
            delta_p[index-num_types*2]   += -1
            delta_p[index+1-num_types*2] +=  1
            print('mutation from p_', index-num_types*2+1, ' to p_', index+2-num_types*2)
        elif index == num_types * 3 - 1:
            delta_p[-1]                  += -1
            delta_p[0]                   +=  1
            print('mutation from p_', num_types, ' to p_', 1)

        # parasite mutation i --> i-1 backward
        elif index == num_types * 3:
            delta_p[0]       += -1
            delta_p[-1]      +=  1
            print('mutation from p_', 1, ' to p_', num_types)
        else:
            delta_p[index-num_types*3]   += -1
            delta_p[index-num_types*3-1] +=  1
            print('mutation from p_', index-num_types*3+1, ' to p_', index-num_types*3)    
    # print(delta_t)
    return delta_h, delta_p, delta_t

