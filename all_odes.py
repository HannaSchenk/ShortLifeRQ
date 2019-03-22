import numpy as np
from scipy.integrate import odeint

# method:

# local_update (discrete time)  0
# moran (discrete time)         1
# gillespie_local_update        2
# gillespie_moran               3

# song_JTB (4D)                 4
# logistic_Lotka_Volterra (4D)  5

def integrate_2d(method, h0_ode, p0_ode, time_ode, w_H, w_P, Mh, Mp, N_H, N_P):

    if method in ['local_update']:
        def deriv_local_update(y_ode, t):
            h_ode = y_ode[0]
            p_ode = y_ode[1]
            pi_H     = np.dot(Mh, np.array([p_ode, 1 - p_ode]))  # matrix * vector = vector
            pi_h_avg = np.dot(pi_H, np.array([h_ode, 1 - h_ode]))  # vector*vector dotproduct = scalar
            pi_P     = np.dot(Mp, [h_ode, 1 - h_ode])
            pi_p_avg = np.dot(pi_P, [p_ode, 1 - p_ode]) 
            hdot = h_ode * (pi_H - pi_h_avg) * w_H / N_H  # discrete time (not gillespie): divide by N_H
            pdot = p_ode * (pi_P - pi_p_avg) * w_P / N_P
            return np.array([hdot[0], pdot[0]])
        y_out = odeint(deriv_local_update,np.concatenate([h0_ode, p0_ode]),time_ode)

    elif method in ['gillespie_local_update']:
        def deriv_local_update(y_ode, t):
            h_ode = y_ode[0]
            p_ode = y_ode[1]
            pi_H     = np.dot(Mh, np.array([p_ode, 1 - p_ode])) 
            pi_h_avg = np.dot(pi_H, np.array([h_ode, 1 - h_ode])) 
            pi_P     = np.dot(Mp, [h_ode, 1 - h_ode])
            pi_p_avg = np.dot(pi_P, [p_ode, 1 - p_ode]) 
            hdot = h_ode * (pi_H - pi_h_avg) * w_H
            pdot = p_ode * (pi_P - pi_p_avg) * w_P
            return np.array([hdot[0], pdot[0]])
        y_out = odeint(deriv_local_update, np.concatenate([h0_ode, p0_ode]), time_ode)
    
    elif method in ['moran']:
        def deriv_moran(y_ode, t):
            h_ode = y_ode[0]
            p_ode = y_ode[1]
            pi_H     = np.dot(Mh, np.array([p_ode, 1 - p_ode]))
            pi_h_avg = np.dot(pi_H, np.array([h_ode, 1 - h_ode]))  
            pi_P     = np.dot(Mp, [h_ode, 1 - h_ode])
            pi_p_avg = np.dot(pi_P, [p_ode, 1 - p_ode]) 
            hdot = h_ode * (pi_H - pi_h_avg) * w_H / (1 - w_H + pi_h_avg * w_H) / N_H  # discrete time (not gillespie): divide by N_H
            pdot = p_ode * (pi_P - pi_p_avg) * w_P / (1 - w_P + pi_p_avg * w_P) / N_P
            return np.array([hdot[0], pdot[0]])
        y_out = odeint(deriv_moran, np.concatenate([h0_ode, p0_ode]), time_ode)
    
    elif method in ['gillespie_moran']:
        def deriv_moran(y_ode, t):
            h_ode = y_ode[0]
            p_ode = y_ode[1]
            pi_H     = np.dot(Mh, np.array([p_ode, 1 - p_ode])) 
            pi_h_avg = np.dot(pi_H, np.array([h_ode, 1 - h_ode]))  
            pi_P     = np.dot(Mp, [h_ode, 1 - h_ode])
            pi_p_avg = np.dot(pi_P, [p_ode, 1 - p_ode]) 
            hdot = h_ode * (pi_H - pi_h_avg) * w_H / (1 - w_H + pi_h_avg * w_H)
            pdot = p_ode * (pi_P - pi_p_avg) * w_P / (1 - w_P + pi_p_avg * w_P)
            return np.array([hdot[0], pdot[0]])
        y_out = odeint(deriv_moran, np.concatenate([h0_ode, p0_ode]), time_ode)

    elif method in ['song_JTB']:
        def deriv_song(y_ode, t):
            h_ode = y_ode[0]
            p_ode = y_ode[1]
            pi_H     = np.dot(Mh, np.array([p_ode, 1 - p_ode])) 
            pi_h_avg = np.dot(pi_H, np.array([h_ode, 1 - h_ode])) 
            pi_P     = np.dot(Mp, [h_ode, 1 - h_ode])
            pi_p_avg = np.dot(pi_P, [p_ode, 1 - p_ode]) 
            hdot = h_ode * (pi_H - pi_h_avg) * w_H
            pdot = p_ode * (pi_P - pi_p_avg) * w_P
            return np.array([hdot[0], pdot[0]])
        y_out = odeint(deriv_song, np.concatenate([h0_ode, p0_ode]), time_ode)

    return y_out[:,0] * N_H, y_out[:,1] * N_P


def integrate_4d_LV(h0_ode, h20_ode, p0_ode, p20_ode, time_ode, birth_H_LV, death_P_LV, mu, lam):
    def deriv_LV(y_ode, t):
        h1_ode = y_ode[0]
        h2_ode = y_ode[1]
        p1_ode = y_ode[2]
        p2_ode = y_ode[3]
        h1_dot = birth_H_LV * h1_ode * (1 - (h1_ode + h2_ode) * (mu / birth_H_LV)) - lam * h1_ode * p1_ode
        h2_dot = birth_H_LV * h2_ode * (1 - (h1_ode + h2_ode) * (mu / birth_H_LV)) - lam * h2_ode * p2_ode
        p1_dot = p1_ode * (lam * h1_ode - death_P_LV)
        p2_dot = p2_ode * (lam * h2_ode - death_P_LV)
        return np.array([h1_dot, h2_dot, p1_dot, p2_dot])
    y_out = odeint(deriv_LV, np.concatenate([h0_ode, h20_ode, p0_ode, p20_ode]), time_ode)
    return y_out[:,0], y_out[:,1], y_out[:,2], y_out[:,3]