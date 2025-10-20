# Make my own Wiener Filter in pixel space

from cmb import CMB 
import jax
import jax.numpy as jnp 
import healpy as hp 
from utils import load_config
import nifty.re as jft 
import numpy as np
import camb
from utils import Dl_to_Cl
from jax import random
import matplotlib.pyplot as plt

c = load_config()

from jaxbind.contrib.jaxducc0 import get_healpix_sht



pars = camb.set_params(H0=67.72,
            ombh2=0.022,
            omch2=0.1192,
            mnu=0.06,
            omk=0,
            tau=0.054,
            As=2.09e-9,
            ns=0.9667,
            halofit_version="mead",
            lmax=2 * c['nside'],
        )
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit = 'K', lmax=2 * c['nside'])
power_spec = powers["total"]
TTCl = Dl_to_Cl(power_spec[:, 0]) 

key = jax.random.PRNGKey(c['seed'])
key, subkey = jax.random.split(key)

class FixedPowerCorrelatedField(jft.Model):
    def __init__(self, nside, c, C_ell):
        self.c = c
        self.nside = nside
        distances = 1.
        self.grid = jft.correlated_field.make_grid(
            nside, distances=distances, harmonic_type="spherical"
        )
        self.C_ell = C_ell
        super().__init__(domain=jax.ShapeDtypeStruct(shape=self.grid.harmonic_grid.shape, dtype=jnp.float64))

    def amplitude_spectrum(self):
        return jnp.sqrt(self.C_ell/2)

    def correlate(self, x):
        sht = get_healpix_sht(
                nside=self.nside,
                nthreads=1,
                lmax=2 * self.nside,
                mmax=2 * self.nside,
                spin=0,
            )
        harmonic_dvol = 1 / self.grid.total_volume
        a = self.amplitude_spectrum()
        a = a[:self.grid.harmonic_grid.lmax + 1]
        a = a[self.grid.harmonic_grid.power_distributor]

        input_T = a * x
        # i = jnp.array(sht(input_T))
        input_T = jnp.reshape(input_T, (1, jnp.shape(input_T)[0]))

        return jnp.array(sht(input_T))[0][0]

    def __call__(self, x):
        return self.correlate(x)


nside = 128
cmb_model = FixedPowerCorrelatedField(nside, c= c, C_ell = TTCl)




cmb_input = jft.random_like(key, cmb_model.domain)
cmb_field = cmb_model(cmb_input)

nstd = 1e-4
noise_cov = lambda x: nstd**2 * x
noise_cov_inv = lambda x: nstd**-2 * x
noise_1 = (
    (noise_cov(jft.ones_like(cmb_model.target))) ** 0.5
) * jft.random_like(key, cmb_model.target)

all_data = cmb_field + noise_1


lh = jft.Gaussian(all_data, noise_cov_inv).amend(cmb_model)
pos_init = jft.Vector(jft.random_like(key, cmb_model.domain))
delta = 1e-6
key, k_w = random.split(key)
samples, info = jft.wiener_filter_posterior(
    lh,
    key=k_w,
    n_samples=1,
    draw_linear_kwargs=dict(
        cg_name="W",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
)

post_mean, post_std = jft.mean_and_std(tuple(cmb_model(s) for s in samples))

to_plot = [
    ("Signal", cmb_field),
    ("Noise", noise_1),
    ("Data", all_data),
    ("Posterior Mean", post_mean),
    ("Posterior Standard Deviation", post_std),
]
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 20),
                         subplot_kw={'projection': 'mollweide'})
axes = axes.flatten()

if len(axes) > len(to_plot):
    for ax in axes[len(to_plot):]:
        ax.remove()

for ax, (title, field) in zip(axes, to_plot):
    hp.mollview(field * 1e6, norm='hist', title=title, unit=r'$\mu K$', fig = fig, sub=(n_rows, n_cols, list(axes).index(ax)+1))
    ax.set_axis_off()

plt.figtext(
    0.88, 0.02,  # (x, y) in figure coordinates (0–1)
    f"Noise σ = {nstd} μK\n",
    ha='right', va='bottom', fontsize=16, color='black'
)

plt.tight_layout()
plt.savefig('WF.pdf', dpi = 500)
plt.show()