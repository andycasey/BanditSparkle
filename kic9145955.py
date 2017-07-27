
import os
import itertools
import numpy as np
from collections import OrderedDict
from astropy.table import Table
from time import time

import cadence
from model import AsteroseismicModel


FIGURES_PATH = "figures_20170720"

if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)

def savefigs(fig, basename):
    fig.savefig("{}/{}.png".format(FIGURES_PATH, basename), dpi=300)
    fig.savefig("{}/{}.pdf".format(FIGURES_PATH, basename), dpi=300)


object_name = "KIC 9145955"
nu_max, nu_max_error = (129.69e-6, 2.52e-6) # [Hz]
delta_nu, delta_nu_error = (10.93e-6, 0.23e-6) # [Hz]


apokasc_value = {
    "nu_max": (nu_max, nu_max_error),
    "delta_nu": (delta_nu, delta_nu_error)
}

power_spectrum = Table.read("KIC009145955_d22_v1.fits")


light_curve = Table.read("KIC009145955_d21_v1.fits")
light_curve["TIME"] -= np.min(light_curve["TIME"]) 
light_curve["TIME"] *= 24 * 60.* 60 # convert date tto seconds
light_curve["FLUX"] *= 1e-12 # convert to mags

model = AsteroseismicModel()

t = cadence.tess()
y = model.predict_light_curve(t, nu_max, delta_nu)
omega, ft = model.predict_power_spectrum(t, nu_max, delta_nu)


fig, axes = plt.subplots(3)
axes[0].plot(t, y)
axes[1].plot(omega * 1e6, np.abs(ft)**2)
axes[2].plot(omega, np.angle(ft, deg=True))

# Plot some data.
axes[1].semilogx(
    power_spectrum["FREQUENCY"], power_spectrum["PSD"],
    c='k', zorder=-1)

fig.tight_layout()
savefigs(fig, object_name)



# Evaluate log-likelihood for ~5000 Kepler points
N = 1000 # Approximate number of points.
S = int(float(len(light_curve))/N)
t_sparse_kepler = light_curve["TIME"][::S]
y_sparse_kepler = light_curve["FLUX"][::S]
yerr_sparse_kepler = 1e-5 * np.ones(len(t_sparse_kepler))

# Get time series for gaia.
t_gaia = cadence.gaia("KIC12008916_gaia_forecast.csv")
idx_gaia = np.ones(t_gaia.size, dtype=int)
keep = np.ones(t_gaia.size, dtype=bool)
for i, t_ in enumerate(t_gaia):

    diffs = np.diff(np.abs(t_ - light_curve["TIME"]))
    idx = np.argmin(diffs)
    # Only keep if less than 30 minute differences.
    if diffs[idx] < 30 * 60 * 60:
        idx_gaia[i] = idx

    else:
        keep[i] = False

print("Dropped {} observations as bad cadences".format(sum(~keep)))

t_gaia = t_gaia[keep]
idx_gaia = idx_gaia[keep]
y_gaia = light_curve["FLUX"][idx_gaia]
yerr_gaia = 1e-5 * np.ones(len(y_gaia))

cadence_name = "sparse_kepler"

data = {
    "gaia": (t_gaia, y_gaia, yerr_gaia),
    "sparse_kepler": (t_sparse_kepler, y_sparse_kepler, yerr_sparse_kepler)
}

t, y, yerr = data[cadence_name]


fiducial_value = OrderedDict([
    ("nu_0", 129.57e-6),
    ("delta_nu", 10.87e-6),
    ("nu_max", nu_max),
    ("bell_height", None),
    ("bell_width", None), #8 * delta_nu * nu_max),
    ("r_01", 0.5)
])
parameter_names = fiducial_value.keys()


grid_points = {
    "bell_width": np.logspace(-12, -5, 50),
    "bell_height": np.logspace(-30, 5, 50),
}

# Do a grid and select values if necessary.
products = [None] * len(fiducial_value)
for i, (parameter_name, value) in enumerate(fiducial_value.items()):
    if value is None:
        products[i] = grid_points[parameter_name]
    else:
        products[i] = [value]


thetas = np.array([theta for theta in itertools.product(*products)])
P = thetas.shape[0]
ll = -np.inf * np.ones(P)


log_likelihood_cost = 0.75 # seconds per call
print("Expected cost: ~{} seconds ({} theta evaluations @ {} seconds per call"\
    .format(P * log_likelihood_cost, P, log_likelihood_cost))

# Pre-compute required matrices.
model._pre_compute_covariance_matrices(yerr)

# Start evaluations.
t_init = time()
for i, theta in enumerate(thetas):
    ll[i] = model.marginalized_log_likelihood(theta, t, y)

    est_log_likelihood_cost = (time() - t_init)/(i + 1)
    print("{}/{}: {} {} (~{:.0f} secs remaining)".format(
        i, P, theta, ll[i], est_log_likelihood_cost * (P - i + 1)))


# For values on a grid, find the best values and then do nu_max and delta_nu
idx = parameter_names.index("bell_width")
if np.unique(thetas[:, idx]).size > 1:
    # Only take reasonable bell widths
    ok = thetas[:, idx] < 10**(-5.5)
else:
    ok = np.ones(len(thetas), dtype=bool)

_ = np.argmax(ll[np.where(ok)[0]])
ll_idx = np.where(ok)[0][_]

best_theta = thetas[ll_idx]

# Plot stuff.
scales = dict(nu_0=1e6, nu_max=1e6, delta_nu=1e6, bell_height=1e10)

latex_labels = {
    "nu_max": r"$\nu_{\rm max}$ $[\mu{\rm Hz}]$",
    "nu_0": r"$\nu_{0}$ $[\mu{\rm Hz}]$",
    "delta_nu": r"$\Delta\nu$ $[\mu{\rm Hz}]$",
}
for i, (parameter_name, value) in enumerate(fiducial_value.items()):

    if np.unique(thetas[:, i]).size == 1:
        print("Skipping {} because it was fixed".format(parameter_name))
        continue

    fig, ax = plt.subplots()

    ax.set_title("{} with {} cadence ({} obs)".format(
        object_name, cadence_name, len(y)))
    x = thetas[:, i] * scales.get(parameter_name, 1)

    if parameter_name in ("bell_height", "bell_width"):
        x = np.log10(thetas[:, i])

    ax.scatter(x, ll, 
        alpha=0.5, facecolor="k")

    ax.scatter(x[ll_idx], ll[ll_idx], facecolor="r", zorder=100)
    
    ax.set_xlabel(latex_labels.get(parameter_name, parameter_name))
    ax.set_ylabel(r"$\mathcal{L}$")

    fig.tight_layout()
    savefigs(fig, "{}_{}_{}_grid".format(object_name, cadence_name, parameter_name))


# Update the fiducial values.
new_fiducial_value = fiducial_value.copy()
for i, (key, value) in enumerate(new_fiducial_value.items()):
    if value is None:
        print("Updating {} fiducial value from {} to {}".format(key, value, best_theta[i]))
        new_fiducial_value[key] = best_theta[i]



# Plot 1D things.
plot_grids = [
    ("nu_max", np.linspace(1e-6, 260e-6, 260)),
    ("nu_max", np.linspace(nu_max - 10e-6, nu_max + 10e-6, 100)),
    ("delta_nu", np.linspace(10.87e-6 - 0.3e-6, 10.87e-6 + 0.3e-6, 400)),
    ("delta_nu", np.linspace(1e-6, 20e-6, 400))
]

for k, (parameter_name, grid_points) in enumerate(plot_grids):

    P = len(grid_points)
    index = parameter_names.index(parameter_name)
    plot_thetas = np.tile(new_fiducial_value.values(), P).reshape(P, -1)
    plot_thetas[:, index] = grid_points
    plot_ll = -np.inf * np.ones(P)

    for i, theta in enumerate(plot_thetas):
        print(parameter_name, i, P)

        #theta = [each for each in theta]
        #theta[4] = None
        plot_ll[i] = model.marginalized_log_likelihood(theta, t, y)

    scale = scales.get(parameter_name, 1)

    fig, ax = plt.subplots()
    ax.set_title("{} with {} cadence ({} obs)".format(
        object_name, cadence_name, len(y)))

    x = grid_points * scale
    ax.plot(x, plot_ll, c="k", lw=2)

    if apokasc_value.get(parameter_name, None) is not None:
        value, error = apokasc_value[parameter_name]
        ax.axvline(value * scale, alpha=0.9, lw=3, zorder=-1)
        ax.axvspan((value - error) * scale, (value + error) * scale,
            alpha=0.5, zorder=-5)

    ax.set_xlabel(latex_labels.get(parameter_name, parameter_name))
    ax.set_ylabel(r"$\mathcal{L}$")

    fig.tight_layout()

    savefigs(fig, "{}_{}_{}_{}".format(object_name, cadence_name, parameter_name, k))


raise a


