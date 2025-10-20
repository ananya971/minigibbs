import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

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
