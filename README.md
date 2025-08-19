# Frosst et al. (2025)

This repository contains Python code to calculate the tidal forces acting on a mass distribution approximated by a spherical aperture of radius R, as described in Frosst et al. (2025). Intended for use in cosmological simulations, but may be applicable elsewhere.

We use this code in Frosst et al. (2025) to understand how tidal interactions between galaxies leads to bar formation. A full description of the mathematics behind this technique will be available there, upon publication. Only the numpy package is required to run this code, however, we recommend running this function for multiple objects in parallel with, e.g., the multiprocessing package - this will speed up your analysis and is simple to implement.

If you use this code in your work, please cite Frosst et al. (2025).