import nifty.re as jft
import jax.numpy as jnp
from utils import Dl_to_Cl
from jaxbind.contrib.jaxducc0 import get_healpix_sht
import camb
import jax

jax.config.update("jax_enable_x64", True)

class CMB(jft.Model):
    def __init__(self, c):
        self.c = c
        self.nside = c['nside']
        self.distances = c['distances']
        self.nus = jnp.asarray(c['freqs']).reshape(-1, 1)
        self.grid = jft.correlated_field.make_grid(shape = self.nside, distances=self.distances, harmonic_type='spherical')

        self.pars = camb.set_params(H0=67.72,
            ombh2=0.022,
            omch2=0.1192,
            mnu=0.06,
            omk=0,
            tau=0.054,
            As=2.09e-9,
            ns=0.9667,
            halofit_version="mead",
            lmax=self.grid.harmonic_grid.lmax,
        )
        self.results = camb.get_results(self.pars)
        self.powers = self.results.get_cmb_power_spectra(self.pars, CMB_unit = 'K', lmax=self.grid.harmonic_grid.lmax)

        self.power_spec = self.powers["total"]

        self.TTCl = Dl_to_Cl(self.power_spec[:, 0])  # Undo plotting factors
        self.EECl = Dl_to_Cl(self.power_spec[:, 1])
        self.TECl = Dl_to_Cl(self.power_spec[:, 3])
        self.BBCl = Dl_to_Cl(self.power_spec[:, 2])

        # Construct the coeffs of alm's
        self.alm_T = jnp.sqrt(self.TTCl)
        self.alm_T = self.alm_T[
            : self.grid.harmonic_grid.lmax + 1
        ]  # Truncate to nifty's lmax
        self.alm_T = self.alm_T[
            self.grid.harmonic_grid.power_distributor
        ]  # Power distributor

        self.alm_E1 = self.TECl[2:] / jnp.sqrt(self.TTCl[2:])
        self.alm_E1 = jnp.pad(self.alm_E1, (2, 0), constant_values=0) # For monopole and dipole. Replace by 0
        self.alm_E1 = self.alm_E1[: self.grid.harmonic_grid.lmax + 1]
        self.alm_E1 = self.alm_E1[self.grid.harmonic_grid.power_distributor]

        self.alm_E2 = jnp.sqrt((self.EECl[2:]) - ((self.TECl[2:]) ** 2 / (self.TTCl[2:])))
        self.alm_E2 = jnp.pad(self.alm_E2, (2, 0), constant_values=0) # For monopole and dipole. Replace by 0
        self.alm_E2 = self.alm_E2[: self.grid.harmonic_grid.lmax + 1]
        self.alm_E2 = self.alm_E2[self.grid.harmonic_grid.power_distributor]
        #TODO: Change this clipping scheme
        self.alm_B = jnp.sqrt(jnp.clip(self.BBCl / 2, a_min = 0, a_max = None)) # Clipping to avoid small -ve values
        self.alm_B = self.alm_B[: self.grid.harmonic_grid.lmax + 1]
        self.alm_B = self.alm_B[self.grid.harmonic_grid.power_distributor]

        assert not jnp.any(jnp.isnan(self.alm_T)), "alm_T contains NaN"
        assert not jnp.any(jnp.isnan(self.alm_E1)), "alm_E1 contains NaN"
        assert not jnp.any(jnp.isnan(self.alm_E2)), "alm_E2 contains NaN"
        assert not jnp.any(jnp.isnan(self.alm_B)), "alm_B contains NaN"

        # Input domain: dictionary of xi_T, xi_E and xi_B
        super().__init__(
            domain={
                "cmb_tqu": jax.ShapeDtypeStruct(
                    shape=(3, self.grid.harmonic_grid.shape[0]), dtype=jnp.float64
                )
            }
        )

    def __call__(self, xi):
        (
            xi_T,
            xi_E,
            xi_B,
        ) = xi["cmb_tqu"]

        try:
            sht_2 = get_healpix_sht(
                nside=self.nside,
                nthreads=1,
                lmax=2 * self.nside,
                mmax=2 * self.nside,
                spin=2,
            )  # Spin 2 SHT for Q and U
            sht = get_healpix_sht(
                nside=self.nside,
                nthreads=1,
                lmax=2 * self.nside,
                mmax=2 * self.nside,
                spin=0,
            )  # Spin 0 SHT for T
        except Exception as e:
            print(f"get_healpix_sht: {e}")

        qu = jnp.array(
            sht_2(
                jnp.stack(
                    [self.alm_E1 * xi_T + self.alm_E2 * xi_E, self.alm_B * xi_B], axis=0
                )
            )
        )[0]
        qu = qu if 'Q' in self.c['stokes'] else qu.at[0].mul(0.)
        qu = qu if 'U' in self.c['stokes'] else qu.at[1].mul(0.)
        input_T = self.alm_T * xi_T
        input_T = jnp.expand_dims(input_T, 0)
        i = jnp.array(sht(input_T))[0] # Shape modifications for valid input to `sht`
        i = i if 'I' in self.c['stokes'] else jnp.zeros(jnp.shape(i))
        fields = jnp.vstack((i, qu))
        fields = jnp.expand_dims(fields, 1) * jnp.ones(jnp.shape(jnp.expand_dims(self.nus, [0])))
        return fields 






