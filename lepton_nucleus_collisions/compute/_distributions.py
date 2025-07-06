import numpy as np

from .transformations import ENERGY_FRACTION, ENERGY, THETA, COS_THETA, ETA, E_THETA_transform

@auto_integrate
def distribution(experiment, final_state, x = None, y = None, x_var = LOG_GAMMA, y_var = ETA, n_pts_x = 200, n_pts_y = 500, frame = 'lab', final_state_particle = 'boson', from_file = True):

    dcrossx = experiment.differential_cross_section(final_state, x, y, x_var, y_var)
    
    

    v_nuc = experiment.v_nuc
    Ei = experiment.E

    mf = final_state.particle_mass if final_state_particle == 'phi' else final_state_particle.mf

    f = np.sqrt((1 + v_nuc)/(1 - v_nuc)) if frame == 'lab' else 1 # Doppler factor
    E_frame = Ei/f
    
    #GAMMA by default
    x_min = 1.0+1e-3
    x_max = (max(1, E_frame/mf) + mf/(4*E_frame))*1.2 # Empirically, this seems to be an appropriate range for the boost
    
    x = np.geomspace(x_min, x_max, n_pts_x)
    
    if x_var in ENERGY_FRACTION:
        x *= mf/E_frame
    
    if x_var in ENERGY:
        x *= mf
     
    #ETA by default
    y_min = -30
    y_max = 30
    y = np.linspace(y_min, y_max, n_pts_y)
    
    if y_var in THETA:
        y = 2*np.arctan(np.exp(-y))
    
    if y_var in COS_THETA:
        y = 2*np.arctan(np.exp(-y))
        y = np.cos(y)

    X, Y = np.meshgrid(x, y, indexing = 'ij')
    
    #transform the variables
    E, TH = E_THETA_transform(experiment, final_state, X, Y)

    #interpolates automatically
    dcrossx = experiment.differential_cross_section(final_state, E = E, th = TH, from_file = True)
    crossx = experiment.cross_section(final_state, units = 'GeV')

    dist = dcrossx/crossx

    if return_2D:
        return np.stack(X, Y, dist)

    dist_x = np.trapz(dist, x = x, axis = -2)
    dist_y = np.trapz(dist, x = x, axis = -1)

    return np.stack(x, dist_x), np.stack(y, dist_y)
    