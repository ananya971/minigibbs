from jax import numpy as jnp
import json
import os



def Dl_to_Cl(pow_spec):
    pow_specCl = jnp.zeros_like(pow_spec)
    for ell,i in enumerate(pow_spec[2:]):
        c_ell = (jnp.pi * i)/((ell+2)*(ell+3))
        pow_specCl = pow_specCl.at[ell+2].set(c_ell)
    return pow_specCl

def Cl_to_Dl(C_ell):
    pow_specDl = jnp.zeros_like(C_ell)
    for ell,i in enumerate(C_ell[2:]):
        d_ell = ((ell+2)*(ell+3) * i)/(jnp.pi)
        pow_specDl = pow_specDl.at[ell+2].set(d_ell)
    return pow_specDl



def load_config():
    if os.getenv('RUN_CONFIG') == '1':
        config_path = os.getenv('CONFIG_PATH', 'config.json')
    else:
        config_path = 'config.json'
    
    with open(config_path) as f:
        print(f"Using config file: {config_path}")
        return json.load(f)



