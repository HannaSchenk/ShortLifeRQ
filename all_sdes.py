import numpy as np
import sdeint
import math
# method:

# local_update (discrete time)  0
# moran (discrete time)         1
# gillespie_local_update        2
# gillespie_moran               3

# song_JTB (4D)                 4
# logistic_Lotka_Volterra (4D)  5

def stoch_integrate_2d(current_method, sde_method, tmax, delta_t_sde, y0, N_H, N_P, Mh, Mp, w_H, w_P):
    ## initialise
    t_sde = 0
    y_sde = 1. * y0
    yplot = 1. * y0
    tplot = [t_sde]
    # h_sde = y_sde[0]
    # p_sde = y_sde[1]


    if current_method in ['local_update', 'moran', 'gillespie_local_update', 'gillespie_moran']:


        def pi_H1(x):
            return Mh[0, 0] * x + Mh[0, 1] * (1 - x)
        def pi_H2(x):
            return Mh[1, 0] * x + Mh[1, 1] * (1 - x)
        def pi_P1(z):
            return Mp[0, 0] * z + Mp[0, 1] * (1 - z)
        def pi_P2(z):
            return Mp[1, 0] * z + Mp[1, 1] * (1 - z)
    

    if current_method in ['local_update', 'gillespie_local_update']:
        # drift:
        def a_drift(y_sde, t_sde):
            h_sde = y_sde[0]
            p_sde = y_sde[1]
            a1 = h_sde * (1 - h_sde) * w_H * (1 - 2 * p_sde)
            a2 = p_sde * (1 - p_sde) * w_P * (2 * h_sde - 1)
            if current_method == 'local_update':
                a1 = a1 / N_H
                a2 = a2 / N_P
            return np.array([a1, a2])

        # diffusion:
        def b_diffusion(y_sde, t_sde):
            h_sde = y_sde[0]
            p_sde = y_sde[1]
            d1 = h_sde * (1 - h_sde) * 1 / N_H
            d2 = p_sde * (1 - p_sde) * 1 / N_P 
            if current_method == 'local_update':
                d1 = d1 / N_H
                d2 = d2 / N_P
            b1 = math.sqrt(d1)
            b2 = 0
            b3 = 0
            b4 = math.sqrt(d2)
            return np.array([[b1, b2], [b3, b4]])



    if current_method in ['moran', 'gillespie_moran']:
        # drift:
        def a_drift(y_sde, t_sde):
            h_sde = y_sde[0]
            p_sde = y_sde[1]
            pi_h_sde = h_sde * pi_H1(p_sde) + (1 - h_sde) * pi_H2(p_sde)
            pi_p_sde = p_sde * pi_P1(h_sde) + (1 - p_sde) * pi_P2(h_sde)
            a1 = (h_sde * (1 - h_sde) * (pi_H1(p_sde) - pi_H2(p_sde)) * w_H) / (1 - w_P + w_P * pi_h_sde)
            a2 = (p_sde * (1 - p_sde) * (pi_P1(h_sde) - pi_P2(h_sde)) * w_H) / (1 - w_P + w_P * pi_p_sde) 
            if current_method == 'moran':
                a1 = a1 / N_H
                a2 = a2 / N_P
            return np.array([a1, a2])

        # diffusion:
        def b_diffusion(y_sde, t_sde):
            h_sde = y_sde[0]
            p_sde = y_sde[1]
            pi_h_sde = h_sde * pi_H1(p_sde) + (1 - h_sde) * pi_H2(p_sde)
            pi_p_sde = p_sde * pi_P1(h_sde) + (1 - p_sde) * pi_P2(h_sde)
            d1 =  2 * h_sde * (1 - h_sde) * (1 - w_H + w_H * (pi_H1(p_sde) + pi_H2(p_sde)) / 2 ) / ((1 - w_H + w_H * pi_h_sde)) * 1 / N_H
            d2 =  2 * p_sde * (1 - p_sde) * (1 - w_P + w_P * (pi_P1(h_sde) + pi_P2(h_sde)) / 2 ) / ((1 - w_P + w_P * pi_p_sde)) * 1 / N_P
            if current_method == 'moran':
                d1 = d1 / N_H
                d2 = d2 / N_P
            b1 = math.sqrt(d1)
            b2 = 0
            b3 = 0
            b4 = math.sqrt(d2)
            return np.array([[b1, b2], [b3, b4]])

    error_in_sde = 0
    while (t_sde < tmax) and y_sde[0]>0 and y_sde[1]>0 and y_sde[0]<1 and y_sde[1]<1:
        try:
            if sde_method == 'Runge_Kutta':
                y_sde = sdeint.itoSRI2(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            elif sde_method == 'Euler':
                y_sde = sdeint.itoEuler(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            else:
                print('Use other sde_method')
        except ValueError:
            print('Value error in sdeint', ' h, p:', y_sde)
            error_in_sde = 1
            break
        y_sde = y_sde[-1]     
        yplot = np.vstack((yplot, y_sde))
        t_sde += delta_t_sde
        tplot.append(t_sde) 

    return yplot[:, 0] * N_H, yplot[:, 1] * N_P, tplot, error_in_sde

def stoch_integrate_4d_song(sde_method, tmax, delta_t_sde, y0, alpha, beta, w_H, w_P):
    t_sde = 0
    y_sde = 1. * y0
    yplot = 1. * y0.reshape(1,4)
    tplot = [t_sde]
    # drift:
    def a_drift(y_sde, t_sde):
        h_sde = y_sde[:2]
        p_sde = y_sde[2:]
        a_h = h_sde * h_sde[::-1] * (alpha - beta) * (p_sde[::-1] - p_sde) * w_H / (sum(h_sde) * sum(p_sde))
        a_p = p_sde * p_sde[::-1] * (alpha - beta) * (h_sde - h_sde[::-1]) * w_P / (sum(h_sde) * sum(p_sde))
        return np.array([a_h, a_p]).flatten() 

    # diffusion:
    def b_diffusion(y_sde, t_sde):
        h_sde = y_sde[:2]
        p_sde = y_sde[2:]
        d = np.zeros((4, 4)) 
        b = np.zeros((4, 4))
        [d[0, 0], d[1, 1]] = 2 * h_sde * (1 - w_H + w_H * (alpha * (h_sde * p_sde + h_sde[::-1] * sum(p_sde) / 2) + beta * (p_sde[::-1] * h_sde + h_sde[::-1] * sum(p_sde) / 2)))  
        [d[2, 2], d[3, 3]] = 2 * p_sde * (1 - w_P + w_P * (alpha * (p_sde * h_sde + p_sde[::-1] * sum(h_sde) / 2) + beta * (h_sde[::-1] * p_sde + p_sde[::-1] * sum(h_sde) / 2)))
        b[0, 0] = math.sqrt(d[0, 0])
        b[1, 1] = math.sqrt(d[1, 1])
        b[2, 2] = math.sqrt(d[2, 2])
        b[3, 3] = math.sqrt(d[3, 3])
        return b

    error_in_sde = 0
    while (t_sde < tmax) and y_sde[0]>0 and y_sde[1]>0 and y_sde[2]>0 and y_sde[3]>0:
        try:
            if sde_method == 'Runge_Kutta':
                y_sde = sdeint.itoSRI2(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            elif sde_method == 'Euler':
                y_sde = sdeint.itoEuler(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            else:
                print('Use other sde_method')
        except ValueError:
            print('Value error in sdeint', ' h, p:', y_sde)
            error_in_sde = 1
            break
        y_sde = y_sde[-1]
        yplot = np.vstack((yplot, y_sde))
        t_sde += delta_t_sde
        tplot.append(t_sde)
    print(yplot)
    return yplot[:, 0], yplot[:, 1], yplot[:, 2], yplot[:, 3], tplot, error_in_sde

def stoch_integrate_4d_LV(sde_method, tmax, delta_t_sde, h0, h20, p0, p20, bh, dp, mu, lam):

    t_sde = 0
    y0 = np.array([h0, h20, p0, p20]).flatten()
    y_sde = 1. * y0
    yplot = 1. * y0
    tplot = [t_sde]

    # drift:
    def a_drift(y_sde, t_sde):
        h_sde = y_sde[:2]
        p_sde = y_sde[2:]
        a = np.array([bh * h_sde * (1 - mu / (bh) * (sum(h_sde))) - lam * p_sde * h_sde, p_sde * (lam * h_sde - dp)]).flatten() 
        return a

    # diffusion:
    def b_diffusion(y_sde, t_sde):
        h_sde = y_sde[:2]
        p_sde = y_sde[2:]
        d = np.zeros((4, 4)) 
        b = np.zeros((4, 4))
        [d[0, 0], d[1, 1]] = bh * h_sde * (1 + mu / (bh) * (sum(h_sde))) + lam * p_sde * h_sde  # d^2/dh1^2 and d^2/dh2^2  
        [d[2, 2], d[3, 3]] = lam * p_sde * h_sde + dp * p_sde
        b[0, 0] = math.sqrt(d[0, 0])
        b[1, 1] = math.sqrt(d[1, 1])
        b[2, 2] = math.sqrt(d[2, 2])
        b[3, 3] = math.sqrt(d[3, 3])
        return b

    error_in_sde = 0
    while (t_sde < tmax) and y_sde[0]>0 and y_sde[1]>0 and y_sde[2]>0 and y_sde[3]>0:
        try:
            if sde_method == 'Runge_Kutta':
                y_sde = sdeint.itoSRI2(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            elif sde_method == 'Euler':
                y_sde = sdeint.itoEuler(a_drift, b_diffusion, y_sde, [t_sde, t_sde + delta_t_sde])
            else:
                print('Use other sde_method')
        except ValueError:
            print('Value error in sdeint', ' h, p:', y_sde)
            error_in_sde = 1
            break
        y_sde = y_sde[-1]
        yplot = np.vstack((yplot, y_sde))
        t_sde += delta_t_sde
        tplot.append(t_sde)
    return yplot[:, 0], yplot[:, 1], yplot[:, 2], yplot[:, 3], tplot, error_in_sde
