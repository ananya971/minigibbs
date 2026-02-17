#%%
from cmb import CMB
from freefree import FreeFree
from synchrotron import Synchrotron
from utils import load_config, Dl_to_Cl
import jax
import nifty.re as jft
import jax.numpy as jnp
import healpy as hp
import numpy as np
from cg import CG
import camb
from functools import partial
import jaxbind.contrib.jaxducc0 as jaxducc0
from nifty.re import cg
c = load_config()

pars = camb.set_params(
    H0=67.72,
    ombh2=0.022,
    omch2=0.1192,
    mnu=0.06,
    omk=0,
    tau=0.054,
    As=2.09e-9,
    ns=0.9667,
    halofit_version="mead",
    lmax=2 * c["nside"],
)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="K", lmax=2 * c["nside"])
power_spec = powers["total"]
TTCl = Dl_to_Cl(power_spec[:, 0])

key = jax.random.PRNGKey(c["seed"])
key, subkey = jax.random.split(key, num=2)

nstd = 1e-4

ff = FreeFree(
    c=c, file="data/COM_CompMap_freefree-commander_0256_R2.00.fits", Ninv=nstd**-2
)

synch = Synchrotron(
    c=c, file="data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits", Ninv=nstd**-2
)

ff_field = ff.freefree_comm()  # f_j(nu) * f_j (n_freq, n_pix)
synch_field = synch.synch_comm()  # f_j(nu) * f_j (n_freq, n_pix)

cmb_model = CMB(c=c)
input = jft.random_like(jax.random.PRNGKey(c["seed"]), cmb_model.domain)
cmb_field = cmb_model(input)[0, :, :]  # Only choosing Stokes I

noise_truth_1 = (
    jax.random.normal(key, shape=cmb_field.shape, dtype=jnp.float64) * nstd
)  # Z = (X-mu)/sigma where Z is standard normally distributed and X with (mu,sigma)

res = 2 * hp.nside2resol(nside=c["nside"])

data = ff_field + synch_field + cmb_field  # noiseless, sum of all signals
data_alms = [
    hp.map2alm(np.asarray(data)[i], lmax=2 * c["nside"]) for i in range(len(c["freqs"]))
]
data_alms_beam = [
    hp.almxfl(data_alms[i], fl=hp.gauss_beam(fwhm=res, lmax=2 * c["nside"]))
    for i in range(len(c["freqs"]))
]

data_beamed = (
    jnp.asarray(
        [
            hp.alm2map(data_alms_beam[i], nside=c["nside"], lmax=2 * c["nside"])
            for i in range(len(c["freqs"]))
        ]
    )
    + noise_truth_1
)  # added beam, and noise

ff_rhs = ff(data=data_beamed)
synch_rhs = synch(data=data_beamed)

def realalm2alm_jax(alm_real: jnp.ndarray, lmax: int, dtype=jnp.complex128):
    # alm_real: (batch, nreal) float64
    head = alm_real[:, :lmax+1].astype(dtype)  # imag=0 implicitly

    tail = alm_real[:, lmax+1:]                # float64
    # tail must be even-length: pairs (re, im)
    tail2 = tail.reshape(tail.shape[0], -1, 2)
    tail_c = (tail2[..., 0] + 1j * tail2[..., 1]).astype(dtype)
    tail_c = tail_c * (jnp.sqrt(2.0) / 2.0)

    return jnp.concatenate([head, tail_c], axis=1)

cmb_cg = CG(c=c, data=data_beamed, noise=nstd**2, C_ell=TTCl)
cmb_rhs = jaxducc0._alm2realalm(cmb_cg.rhs_cmb().reshape(1,-1), lmax=2 * c["nside"], dtype=jnp.float64)
sht = jaxducc0.get_healpix_sht(nside = c['nside'], lmax = 2 * c['nside'], mmax = 2 * c['nside'], spin = 0, nthreads = 1)

#%%

hp.mollview(np.asarray(sht(cmb_rhs))[0,0], cmap = 'magma', norm = 'hist', title = 'cmb_rhs')
b = jnp.concatenate([cmb_rhs, ff_rhs.reshape(1,-1), synch_rhs.reshape(1,-1)], axis = 1)
# b = {"complex": cmb_rhs, "real": jnp.stack([ff_rhs, synch_rhs])}

signals = jnp.vstack(
    [jnp.expand_dims(ff_field, axis=0), jnp.expand_dims(synch_field, axis=0)]
)

FNF = jnp.einsum("ijk, lji-> kl", signals.T, nstd**-2 * signals)


def adjoint_alm2map(x):
    sht = jaxducc0.get_healpix_sht(
        nside=c["nside"], lmax=2 * c["nside"], mmax=2 * c["nside"], spin=0, nthreads=1
    )
    x = jnp.expand_dims(x, axis=(1, 2))  # Go from [2, npix] to [2,1,1,npix]
    alms = jnp.ones(hp.Alm.getsize(lmax=2 * c["nside"]), dtype=jnp.complex128)
    alms = alms.reshape((1, alms.shape[0]))
    sht_T = jax.linear_transpose(
        sht, jaxducc0._alm2realalm(alms, lmax=2 * c["nside"], dtype=jnp.float64)
    )  # alms proxy for shape, can have any value
    y = np.asarray([sht_T(list(x[i])) for i in range(2)])  # [2, 1, 1, n_realalm]
    y = jnp.squeeze(y, axis=(1))  # [2, 1, n_realalm]
    return jnp.squeeze(
        np.asarray(
            [
                jaxducc0._realalm2alm(y[i], lmax=2 * c["nside"], dtype=jnp.complex128)
                for i in range(2)
            ]
        ),
        axis=1,
    )


ANF = np.squeeze([
    jaxducc0._alm2realalm(hp.almxfl(
        adjoint_alm2map(jnp.sum(nstd**-2 * signals, axis=1))[i],
        fl=hp.gauss_beam(fwhm=res, lmax=2 * c["nside"]),
    ).reshape(1,-1), lmax=2 * c["nside"], dtype=jnp.float64)
    for i in range(len(c["freqs"]))
], axis = 1).T
# TODO: Add jaxducc0._alm2realalm here
ANF = jnp.asarray(ANF)
def FNA(x):
    x = jaxducc0._realalm2alm(x, lmax=2 * c["nside"], dtype=jnp.complex128) 
    bx = hp.almxfl(x[0], fl = hp.gauss_beam(fwhm=res, lmax=2 * c["nside"])).reshape(1,-1)
    bx = jaxducc0._alm2realalm(bx, lmax=2 * c['nside'], dtype=jnp.float64)
    Abx = jaxducc0.get_healpix_sht(nside = c['nside'], lmax = 2 * c['nside'], mmax = 2 * c['nside'], spin = 0)(bx)[0][0]
    # bAx = hp.almxfl(Ax, fl = hp.gauss_beam(fwhm=res, lmax=2 * c["nside"]))
    NbAx = nstd**-2 * Abx
    # FNAx = signals * NbAx
    FNAx = jnp.einsum('ijk, i -> k', signals.T, NbAx).reshape(1,-1)
    return FNAx


def apply_mat(x, Ninv, C_ell):
    """Apply operator"""
    res = 2 * hp.nside2resol(c["nside"])  # [rad]
    beam = hp.gauss_beam(fwhm=res, lmax=2 * c["nside"])
    pwf = hp.pixwin(nside=c["nside"], lmax=2 * c["nside"])
    Al = beam * pwf
    sht = jaxducc0.get_healpix_sht(
        nside=c["nside"], lmax=2 * c["nside"], mmax=2 * c["nside"], spin=0, nthreads=1
    )

    def adjoint_alm2map(x):
        x = list(x.reshape((1, 1, x.shape[0])))
        alms = jnp.ones(hp.Alm.getsize(lmax=2 * c["nside"]), dtype=jnp.complex128)
        alms = alms.reshape((1, alms.shape[0]))
        sht_T = jax.linear_transpose(
            sht, jaxducc0._alm2realalm(alms, lmax=2 * c["nside"], dtype=jnp.float64)
        )
        y = sht_T(x)
        return jaxducc0._realalm2alm(y[0], lmax=2 * c["nside"], dtype=jnp.complex128)

    def apply_A(s, n_freq, Al):
        s_beam = hp.almxfl(alm=s, fl=Al)
        s_beam = s_beam.reshape((1, s_beam.shape[0]))
        s_beam = jaxducc0._alm2realalm(s_beam, lmax=2 * c["nside"], dtype=jnp.float64)
        spix_beam = sht(s_beam)
        return jnp.tile(spix_beam[0][0], n_freq)

    def apply_A_transpose(stacked, n_freq):
        s_summed = stacked.reshape(n_freq, hp.nside2npix(c["nside"])).sum(axis=0)
        spix_summed = adjoint_alm2map(jnp.asarray(s_summed))
        s_beamed = hp.almxfl(alm=spix_summed[0], fl=Al)
        return s_beamed

    x = realalm2alm_jax(x, lmax=2 * c["nside"], dtype=jnp.complex128)
    alm_cl = hp.almxfl(alm=x[0], fl=np.sqrt(C_ell))  # S^(1/2)x
    A_pix_alm = apply_A(alm_cl, c["nfreqs"], Al)
    NA_pix = Ninv * A_pix_alm
    ATNA_pix = apply_A_transpose(NA_pix, c["nfreqs"])
    CATNA_alm = hp.almxfl(ATNA_pix, fl=np.sqrt(C_ell))
    res = jaxducc0._alm2realalm(CATNA_alm + x, lmax=2 * c["nside"], dtype=jnp.float64).reshape(1,-1)
    return res


def block_op(inp, A00, A01, A10, A11):


    len1 = (2 * c['nside'] + 1)**2 # size of alm vector
    len2 =  len(c['components'].keys())
    assert len1 + len2 == len(inp[-1]), 'Size of input is not right'

    y0 = A00(inp[:, :len1]) + jnp.matmul(inp[:, -len2:], A01.T) 
    y1 = A10(inp[:, :len1]) + jnp.matmul(inp[:, -len2:], A11.T)

    res = jnp.concatenate([y0, y1], axis=1)

    return res

apply_mat_part = partial(apply_mat, Ninv=nstd**-2, C_ell=TTCl)

A = partial(block_op, A00=apply_mat_part, A01=ANF, A10=FNA, A11=FNF)

init_sig_alm = hp.synalm(self.C_ell, lmax = 2 * self.c['nside'])

# init_sig = jax.random.normal(key, (2 * c['nside'] + 1)**2 + len(c['components'].keys()), dtype=jnp.float64).reshape(1,-1)

# result_fin = cg(A, b, x0 = init_sig, _raise_nonposdef=False, name="CG", absdelta = 1e-10)[0]
from sample import Cl_given_s_healpy_rep
from plotting import plot_c_ells
from utils import Cl_to_Dl
import matplotlib.pyplot as plt


# map = sht(result_fin[:,:-2])

def gibbs_fg(iter, init_signal, nside):
    smp_C_ell = []
    smp_salm = []
    seed = c['seed']
    key = jax.random.PRNGKey(seed)
    key_i = jax.random.split(key, num = iter)
    signal = init_signal

    for i in range(iter):
        signal = cg(A, b, x0 = signal, _raise_nonposdef = False, name = 'CG')[0]
        result_pix = sht(np.asarray(signal)[:, :-2])
        hp.mollview(result_pix[0][0], norm = 'hist', cmap = 'magma', title = f'{i} result_pix')

        # C_ell = Cl_given_s_healpy(np.asarray(signal_alm), nside = nside) # init_signal in pixel space
        C_ell = Cl_given_s_healpy_rep(np.asarray(signal[:, :-2]), nside = nside, key_i=key_i[i])
        print(f'{key_i[i]= }')
        plt.figure()
        plot_c_ells(Cl_to_Dl(C_ell), label='D_ell check', logx=True,
                    legend=True, show=c['show_plots'], fname=f'D_ell_{i}.png')
        
    if iter-i <=10:
        smp_C_ell.append(C_ell)
        smp_salm.append(signal)

    return smp_C_ell, smp_salm


gibbs_fg(15, init_signal=init_sig, nside = c['nside'])



# How to check for symmetry and positive definiteness of the operator A?


# Try adjoint and symm operator tests
# Ill conditioned?


# x has to have the last two values as real numbers
# def is_positive_definite_operator(A_func, lmax, key=random.PRNGKey(0), num_tests=5):
#     """Check if positive definite."""
#     for i in range(num_tests):
#         key, subkey = random.split(key)
#         a = {
#         "complex_part": jnp.zeros(hp.Alm.getsize(lmax=2 * c["nside"]), dtype=jnp.complex128),
#         "real_part": jnp.zeros(2, dtype=jnp.float64),
#         }
#         x = jft.random_like(subkey, a)
#         xAx = jnp.vdot(x, A_func(x))
#         if xAx <= 0:
#             return False
#     return True


# is_positive_definite_operator(
#     op,
#     lmax = 2 * c['nside'],
#     # domain=jax.ShapeDtypeStruct(
#     #     shape=(
#     #         hp.Alm.getsize(lmax=2 * c["nside"]) + 2,
#     #     ),
#     #     dtype=jnp.complex128,
#     # ),
# )


# %%
