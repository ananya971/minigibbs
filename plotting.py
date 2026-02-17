import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from utils import Dl_to_Cl, load_config, Cl_to_Dl
import camb

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

def plot_map(m, norm='hist', title='', unit='', min=None, max=None,
             show=True, fname=None):
    hp.mollview(m, norm=norm, title=title, unit=unit, min=min, max=max)
    if show:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches='tight')
        plt.close()


def plot_c_ells(c_ell, logx=True, label='', legend=False, show=True,
                fname=None, finished=True):
    ells = np.arange(len(c_ell))
    plt.plot(ells, c_ell, label=label)
    plt.plot(np.arange(len(TTCl)), Cl_to_Dl(TTCl))
    if logx:
        plt.xscale('log')
    if legend:
        plt.legend()
    if finished:
        if show:
            plt.show()
        else:
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
