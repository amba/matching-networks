#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import time
np.set_printoptions(precision=4, linewidth=200)

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

def EFGH_shunt_capacitor(ωvals, C):
    I = np.identity(2, dtype=complex)
    N = ωvals.shape[0]
    M = np.tile(I, (N, 1, 1))
    M[:,1,0] = -1j * ωvals * C
    return M

# def EFGH_series_capcitor(ω, C):
#     return EFGH_series(1/(1j*ω*C))

# def EFGH_shunt_inductor(ω, L):
#     return EFGH_shunt(1j * ω * L)

def EFGH_series_inductor(ωvals, L):
    I = np.identity(2, dtype=complex)
    N = ωvals.shape[0]
    M = np.tile(I, (N, 1, 1))
    M[:,0,1] = -1j* ωvals * L
    return M

# power at Z_l

def gen_matching_network(fvals, *p):
    ωvals = 2 * np.pi * fvals
    if len(p) < 2 or len(p) % 2 != 0:
        raise Exception("gen_matching_network needs even number of args")
    C_L_list = list(p)
    C, L = C_L_list.pop(0), C_L_list.pop(0)
    net = np.matmul(EFGH_series_inductor(ωvals, L), EFGH_shunt_capacitor(ωvals, C))
    while C_L_list:
        C, L = C_L_list.pop(0), C_L_list.pop(0)
        new_net = np.matmul(EFGH_series_inductor(ωvals, L), EFGH_shunt_capacitor(ωvals, C))
        net = np.matmul(new_net, net)
    return net

def load_power_network(Z_s, Z_l, EFGH_matching_network):
    EFGH = np.matmul(EFGH_shunt(Z_l), EFGH_matching_network)
    EFGH = np.matmul(EFGH, EFGH_series(Z_s))
    # source voltage normalized to 1
    V_l = 1/EFGH[:,1,1]
    
    return np.abs(V_l**2 / Z_l)
    

def logspace(start, stop, num_points):
    logs = np.linspace(np.log10(start), np.log10(stop), num_points)
    return 10**logs

i = 0
def optimize_network(f_vals, Z_s, Z_l, N_nets=3):
    def cost_function(p):
        global i
        i = i + 1
        if i % 1000 == 0:
            print("i = ", i)
        network = gen_matching_network(f_vals, *p)
        pvals = load_power_network(Z_s, Z_l, network)
        pvals /= 1/(4*Z_s)
        return np.linalg.norm(1-pvals)
    lower_bound = 1e-3
    upper_bound = 1
    N_args = 2 * N_nets
    bounds = list(zip([lower_bound]*N_args, [upper_bound]*N_args))
    print("bounds: ", bounds)
    return scipy.optimize.dual_annealing(cost_function, bounds=bounds,  #maxiter=5000
                                        # initial_temp=5e4
    )
    
        
#    network =     
# 2 step: 20 -> 4.47 -> 1
# 3 step: 20 -> 7.37 -> 2.71 -> 1
# 4 step: 20 -> 9.46 -> 4.47 -> 2.11 -> 1
# 5 step: 20 -> 11.0 -> 6.03 -> 3.31 -> 1.82 -> 1
factor = 20
Z_s = factor
Z_l = 1
Z_s /= np.sqrt(factor)
Z_l /= np.sqrt(factor)

# Z_m1 = 7.37 / np.sqrt(20)
# Z_m2 = 2.71 / np.sqrt(20)

#f_target = 1


f_vals = logspace(1,3, 100)

t_start = time.time()
res = optimize_network(f_vals,  Z_s, Z_l, N_nets=4)
print("time: ", time.time() - t_start)

print(res)

# C1, L1 = shunt_C_series_L_matching_network(f_target, Z_s, Z_m1)
# C2, L2 = shunt_C_series_L_matching_network(f_target, Z_m1, Z_m2)
# C3, L3 = shunt_C_series_L_matching_network(f_target, Z_m2, Z_l)
# print("net1: ", C1, L1)
# print("net2: ", C2, L2)
# print("net3: ", C3, L3)
# # #for i in range(1,1000):
# #  #   print(i)

network = gen_matching_network(f_vals, *res.x)
# #  )

p_vals = load_power_network(Z_s, Z_l, network)
p_vals /= 1/(4*Z_s)
plt.plot(f_vals, np.sqrt(p_vals), "x")
#plt.ylabel("insertion loss (dB)")
plt.xlabel("f/f_0 (Hz)")
plt.xscale("log")

plt.grid()

plt.show(block=False)
import code
code.interact()



