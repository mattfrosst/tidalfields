# -*- coding: utf-8 -*-
"""
@file      calculate_tidalfield.py
@author    Matthew Frosst
@copyright Matthew Frosst (2025)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
           see file LICENSE for details

@version   0.0   jan-2025 MF  initial code
@version   0.1   aug-2025 MF  calculate_tidalfield.py
"""
version = 0.1

import numpy as np
    
###############################################################################################

def tidal_index(x, m, R):
    """
    Calculate the tidal index for a galaxy and its environment.

    Parameters:
    x : ndarray
        An n-by-3 array of Cartesian coordinates of particles, with the origin at the galaxy center.
    m : ndarray
        A 1D array of masses for each particle.
    R : float
        Spherical radius separating the galaxy from its environment.

    Returns:
    dict : 
        A dictionary containing:
        - 'ta': The scalar tidal parameter `ta`. <-- recommended metric.
        - 'tb': The scalar tidal parameter `tb`.
        - 'M': The mass of particles within radius R.
    """

    # Calculate squared distances from the origin
    rsqr = x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2

    # Identify particles within and outside the radius R
    i = np.where(rsqr <= R**2)[0]  # indices of particles in the system
    j = np.where(rsqr > R**2)[0]   # indices of particles in the environment

    # Calculate the mass of particles within radius R
    M = np.sum(m[i])

    # Calculate the tidal tensor due to the environment
    w = np.sqrt(m[j])[:, np.newaxis] * (rsqr[j]**(-5/4))[:, np.newaxis] * x[j, :]
    tidal_tensor = (R**3 / M) * (3 * np.dot(w.T, w) - np.sum(m[j] / rsqr[j]**(3/2)) * np.eye(3))

    try:
        (eigenvals, eigenvects) = np.linalg.eig(tidal_tensor)
        order = np.flip(np.argsort(eigenvals))
        eigenvals  = eigenvals[order]
        eigenvects = eigenvects[:, order]
    
        # Calculate `ta` and `tb`
        ta = np.sqrt(np.sum(np.diag(np.dot(tidal_tensor, tidal_tensor))) / 6) / np.sqrt(1/6)
        tb = (np.diff([eigenvals.min(), eigenvals.max()])[0] / 3) / np.sqrt(1/6)
        
        return (ta, tb, M)
    except (np.linalg.LinAlgError) as e:
        print('LinAlgError: Array must not contain infs or NaNs', e)
        return (np.nan, np.nan, np.nan)
    
###############################################################################################

def example_use():
    """
    Demonstrates the usage of the `tidal_index` function with a simple two-halo test case.
    --- In no way is this a realistic senario. Use at your own risk. ---
    
    The setup includes:
    - A central spherical distribution of 1000 particles centered at (0, 0, 0),
      with a total mass of 1.0 (each particle has mass 1e-3).
    - A secondary spherical distribution of 100 particles centered at (2, 2, 2),
      with total mass 0.1 (each particle has mass 1e-3).
    
    The tidal index is computed for all particles within a fixed radius R.
    """
    np.random.seed(42)  # For reproducibility

    # Parameters
    N_primary = 1000
    N_secondary = 100
    m_particle = 1e-3
    R = 1.0  # Radius for tidal index calculation

    # Primary spherical distribution at (0, 0, 0)
    x_primary = np.random.normal(0, 0.5, size=(N_primary, 3))

    # Secondary spherical distribution at (5, 5, 5)
    x_secondary = np.random.normal(0, 0.5, size=(N_secondary, 3)) + np.array([2, 2, 2])

    # Combine positions and masses
    x_all = np.vstack([x_primary, x_secondary])
    m_all = np.full(x_all.shape[0], m_particle)
    
    # Call the tidal index function
    tidal_values = tidal_index(x_all, m_all, R)
    
    # Print basic stats
    print("Tidal Index Stats:")
    print(f" - T_a: {tidal_values[0]:.4f}")
    print(f" - T_b: {tidal_values[1]:.4f}")
    print(f" - M_enclosed: {tidal_values[2]:.4f}")

    # Optionally return the values for inspection
    return tidal_values

###############################################################################################

if __name__ == "__main__":
    # run example function
    example_use()
