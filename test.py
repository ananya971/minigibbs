import jax 
import healpy as hp 
import camb
import nifty.re as jft
from cmb import CMB
from utils import load_config
from sample import gibbs
import numpy as np
from utils import Dl_to_Cl
import jax.numpy as jnp
from plotting import plot_map

c = load_config()

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

seed = c['seed']

key = jax.random.PRNGKey(seed)
cmb_model = CMB(c=c)

input = jft.random_like(key, cmb_model.domain)

cmb_field = cmb_model(input)[0][0]

key, subkey = jax.random.split(key, num= 2)

nstd = 1e-4

noise_truth_1 = (jax.random.normal(key, shape = cmb_field.shape, dtype = jnp.float64) * nstd) # Z = (X-mu)/sigma where Z is standard normally distributed
noise_truth_2 = (jax.random.normal(subkey, shape = cmb_field.shape, dtype = jnp.float64) * nstd)

plot_map(noise_truth_1*1e6, norm='hist', title=f'noise_truth_1', unit='uK',
         show=c['show_plots'], fname=f'noise_truth_1.png')
plot_map(noise_truth_2*1e6, norm='hist', title=f'noise_truth_2', unit='uK',
         show=c['show_plots'], fname=f'noise_truth_2.png')
cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])
plot_map(hp.alm2map(cmb_alms, nside=c['nside'])*1e6, norm='hist',
         title=f'cmb_input', unit='uK', show=c['show_plots'],
         fname=f'cmb_input.png')

data_1 = hp.alm2map(hp.almxfl(cmb_alms, fl = hp.gauss_beam(2 * hp.nside2resol(c['nside']))), nside = c['nside'], lmax = 2 * c['nside']) + noise_truth_1 # d_1 = A(s) + n_1

plot_map(data_1*1e6, norm='hist', title=f'data_1', unit='uK',
         show=c['show_plots'], fname=f'data_1.png')
data_2 = hp.alm2map(hp.almxfl(cmb_alms, fl = hp.gauss_beam(2 * hp.nside2resol(nside = c['nside']))), nside = c['nside'], lmax = 2 * c['nside']) + noise_truth_2 # d_2 = A(s) + n_2
plot_map(data_2*1e6, norm='hist', title=f'data_2', unit='uK',
         show=c['show_plots'], fname=f'data_2.png')

all_data = np.append(data_1, data_2)
all_noise = np.append(noise_truth_1, noise_truth_2)

assert all_data.shape == all_noise.shape, "Data and noise shapes do not match!"

# Sample from C_ell

cmb_alms = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside'], mmax = 2 * c['nside'])


result_ps, result_alm = gibbs(iter = 5, init_ps= TTCl, data = all_data, noise = jnp.ones(2*hp.nside2npix(c['nside']))*(nstd**2), nside = c['nside']) 


# Checking power spectra of data vs. cmb
# import matplotlib.pyplot as plt 

# data1_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(data_1), lmax = 2 * c['nside']), lmax = 2*c['nside'])

# plt.plot(jnp.arange(data1_cl.shape[0]), data1_cl, label = 'data_1')

# data2_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(data_2), lmax = 2 * c['nside']), lmax = 2*c['nside'])

# noise1_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(noise_truth_1), lmax = 2 * c['nside']), lmax = 2 * c['nside'])

# noise2_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(noise_truth_2), lmax = 2 * c['nside']), lmax = 2 * c['nside'])

# plt.plot(jnp.arange(data2_cl.shape[0]), data2_cl, label = 'data_2')
# cmb_cl = hp.alm2cl(alms1 = hp.map2alm(np.asarray(cmb_field), lmax = 2 * c['nside']), lmax = 2*c['nside'])
# plt.plot(jnp.arange(cmb_cl.shape[0]), cmb_cl, label = 'cmb')


# plt.plot(jnp.arange(noise1_cl.shape[0]), noise1_cl, label = 'noise_1')

# plt.plot(jnp.arange(noise2_cl.shape[0]), noise2_cl, label = 'noise_2')

# plt.xscale('log')

# plt.legend()

# hp.mollview(data_1*1e6, norm = 'hist')









