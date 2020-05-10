#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# return (C, L)
# R_s, R_l must be real for now
def shunt_C_series_L_matching_network(f, R_s, R_l):
    ω = 2 * np.pi * f
    if R_s < R_l:
        raise Exception("Rs must be larger than Rl")
    L = 1/ω * np.sqrt(R_l * (R_s - R_l))
    C = L / (R_l**2 + ω**2*L**2)
    return (C, L)

#
#  V1----> I1 ->--|----------|---> I2 ->---V2 
#                 | two-port |
#  GND------------|----------|-------------

# we call the params E, F, G, H due to different convention from
# usual ABCD params:

#  ( V2 )   ( E   F ) ( V1 )
#  (    ) = (       ) (    )
#  ( I2 )   ( G   H ) ( I1 )



# series impedance (no connection to ground)

#         - - Z - - 
#        |         |
#                 
#        |         |
def EFGH_series(Z):
    EFGH = np.identity(2, dtype=complex);
    EFGH[0,1] = -Z;
    return EFGH

# single shunt impedance

#   --------------     
#       |            
#     Z_shunt_1     
#       |

def EFGH_shunt(Z):
    EFGH = np.identity(2, dtype=complex)
    EFGH[1,0] = -1/Z
    return EFGH

def EFGH_shunt_capacitor(ω, C):
    return EFGH_shunt(1/(1j*ω*C))

def EFGH_series_capcitor(ω, C):
    return EFGH_series(1/(1j*ω*C))

def EFGH_shunt_inductor(ω, L):
    return EFGH_shunt(1j * ω * L)

def EFGH_series_inductor(ω, L):
    return EFGH_series(1j * ω * L)
    

def EFGH_C_shunt_L_series_matching_network(ω, C, L):
    return EFGH_series_inductor(ω,L).dot(EFGH_shunt_capacitor(ω, C))

def EFGH_dual_C_shunt_L_series_matching_network(ω, C1, L1, C2, L2):
    net2 = EFGH_C_shunt_L_series_matching_network(ω, C2, L2)
    return net2.dot(EFGH_C_shunt_L_series_matching_network(ω, C1, L1))

# load power at 
def load_power_network(Z_s, Z_l, EFGH_matching_network):
    EFGH = EFGH_shunt(Z_l).dot(EFGH_matching_network).dot(EFGH_series(Z_s))
    EFGH_inv = np.linalg.inv(EFGH)
    # source voltage normalized to 1
    V_l = 1/EFGH_inv[0,0]
    return np.abs(V_l**2 / Z_l)
    
                                                          
    

def load_power_network_vals(ω_vals, Z_s, Z_l, C, L):
    p_vals = []
    for ω in (ω_vals):
        matching_network = EFGH_C_shunt_L_series_matching_network(ω, C, L)
        p = load_power_network(Z_s, Z_l, matching_network)
        p_vals.append(p)
    return np.array(p_vals)
    

def dual_load_power_network_vals(ω_vals, Z_s, Z_l, C1, L1, C2, L2):
    p_vals = []
    for ω in (ω_vals):
        matching_network = EFGH_dual_C_shunt_L_series_matching_network(ω, C1, L1, C2, L2)
        p = load_power_network(Z_s, Z_l, matching_network)
        p_vals.append(p)
    return np.array(p_vals)

# 2 step: 1000 -> 223 -> 50
# 3 step: 1000 -> 368 -> 136 -> 50
# 4 step: 1000 -> 473 -> 223 -> 106 -> 50
# 5 step: 1000 -> 549 -> 302 -> 166 -> 91 -> 50
Z_s = 1000
Z_l = 50
Z_m = 220
f_target = 50e6
span = f_target / 2

C1, L1 = shunt_C_series_L_matching_network(f_target, Z_s, Z_m)
C2, L2 = shunt_C_series_L_matching_network(f_target, Z_m, Z_l)
 
f_vals = np.linspace(f_target+span, f_target-span, 1000)
ω_vals = 2 * np.pi * f_vals
p_vals = dual_load_power_network_vals(ω_vals, Z_s, Z_l, C1, L1, C2, L2)
p_vals /= 1/(4*Z_s)
plt.plot(f_vals, 10 * np.log10(p_vals))
plt.ylabel("insertion loss (dB)")
plt.xlabel("f (Hz)")
#plt.xscale("log")

plt.grid()

plt.show(block=False)
import code
code.interact()



