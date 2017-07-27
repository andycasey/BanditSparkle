
"""
KIC 12020797: An APOKASC upper red giant branch star.

From Table 3 of Pinsonneault et al. (2014):
Teff = (4311, 65)
logg = (1.512, 0.02)
[Fe/H] = (-0.32, 0.07)

Dnu = (0.88, 0.03) \mu Hz
nu_max = (4.44, 0.27) \mu Hz

M1 = (0.89, 0.22)
R1 = (27.73, 2.36)

NOTES: KASC, PUBLIC, LUMINOUS_RGB, 
"""

import numpy as np


import cadence
from model import AsteroseismicModel




amplitudes = 1e-6
nu_max, nu_max_error = (4.44e-6, 0.27e-6) # [Hz]
delta_nu, delta_nu_error = (0.88e-6, 0.03e-6) # [Hz]


model = AsteroseismicModel()

t = cadence.gaia("")
y = model.predict_light_curve(t, nu_max, delta_nu, amplitudes=amplitudes)
omega, ft = model.predict_power_spectrum(t, nu_max, delta_nu, amplitudes=amplitudes)





fig, axes = plt.subplots(3)
axes[0].plot(t, y)
axes[1].semilogx(omega, np.abs(ft)**2)
axes[2].semilogx(omega, np.angle(ft, deg=True))




raise a