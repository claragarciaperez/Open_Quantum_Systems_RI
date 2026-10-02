#%%
"""
This code simulates the time evolution of a oquantum systems (simple quantum oscillator with frequancy ws),
where the bath is given by an ancilla system (simple quantum oscillator with frequancy wa) in a Gibbs
thermal state at temperature T=1/beta. The interaction between the system and the ancilla is given by a Hamiltonian with interaction terms Jxx, Jyy and Jzz. 

H = -ws/2 * sigma_z(S)*I - wa/2 * I*sigma_z(A) + Jxx * sigma_x(S) * sigma_x(A) + Jyy * sigma_y(S) * sigma_y(A) + Jzz * sigma_z(S) * sigma_z(A)

It is simulated using the repeated interaction protocol, where the system interacts with a fresh ancilla at each time step tau. 
The evolution of the system's state is calculated by tracing out the ancilla after each interaction.

https://doi.org/10.1088/2058-9565/adc7d4

"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def sigma_x():
    return np.array([[0,1],[1,0]])
def sigma_y():
    return np.array([[0,-1j],[1j,0]])
def sigma_z():
    return np.array([[1,0],[0,-1]])

def H_tot_calculation_direct(Jxx,Jyy,Jzz,omega_A,omega_S):
    return np.array([[Jzz - omega_A/2-omega_S/2,0,0,Jxx-Jyy],[0,-Jzz + omega_A/2-omega_S/2,Jxx + Jyy,0],[0,Jxx + Jyy,-Jzz - omega_A/2+omega_S/2,0],[Jxx-Jyy,0,0,Jzz + omega_A/2+omega_S/2]])

def H_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S):
    H_S = -omega_S/2*sigma_z()
    H_A = -omega_A/2*sigma_z()
    H_I = Jxx*np.kron(sigma_x(),sigma_x()) + Jyy*np.kron(sigma_y(),sigma_y()) + Jzz*np.kron(sigma_z(),sigma_z())
    I = np.eye(2)
    return np.kron(H_S, I) + np.kron(I, H_A) + H_I

def U_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S,tau):
    theta = np.sqrt(4*(Jxx+Jyy)**2 + (omega_A - omega_S)**2)
    phi = np.sqrt(4*(Jxx-Jyy)**2 + (omega_A + omega_S)**2)
    S_phi = np.sin(phi*tau/2)
    S_theta = np.sin(theta*tau/2)
    C_phi = np.cos(phi*tau/2)
    C_theta = np.cos(theta*tau/2)
    U = np.array([[np.exp(-1j*Jzz*tau)*(C_phi + 1j*(omega_A + omega_S)*S_phi/phi),0,0,-2*1j*np.exp(-1j*Jzz*tau)*(Jxx-Jyy)*S_phi/phi],
                  [0,np.exp(1j*Jzz*tau)*(C_theta + 1j*(omega_A - omega_S)*S_theta/theta),-2*1j*np.exp(1j*Jzz*tau)*(Jxx+Jyy)*S_theta/theta,0],
                  [0,-2*1j*np.exp(1j*Jzz*tau)*(Jxx+Jyy)*S_theta/theta,np.exp(1j*Jzz*tau)*(C_theta + 1j*(omega_A - omega_S)*S_theta/theta),0],
                  [-2*1j*np.exp(-1j*Jzz*tau)*(Jxx-Jyy)*S_phi/phi,0,0,np.exp(-1j*Jzz*tau)*(C_phi - 1j*(omega_A + omega_S)*S_phi/phi)]])
    return U

def rho_S_calculation(ps,cs):
    return np.array([[ps,cs],[np.conj(cs),1-ps]])


def partial_trace(rho, dims, subsystem):
    dA, dB = dims
    rho = rho.reshape(dA, dB, dA, dB)

    if subsystem == 0:  # trazar A
        return np.trace(rho, axis1=0, axis2=2)
    elif subsystem == 1:  # trazar B
        return np.trace(rho, axis1=1, axis2=3)


#%%
omega_S = 1 #Frequency of the system
omega_A = 2 #Frequency of the ancilla
Jxx = 2 #Interaction term in the Hamiltonian x
Jyy = 1 #Interaction term in the Hamiltonian y
Jzz = 0 #Interaction term in the Hamiltonian z
beta = 1 #Inverse temperature
n = 1000000 #Number of iterations
#%%
H_tot =H_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S)
p_A = 1/(1+np.exp(-beta*omega_A))
rho_A = np.array([[p_A,0],[0,1-p_A]])
#%% Evolution of rhoS for different values of tau
ps = []
cs= []
tau = [0.001,0.01,0.1,1,10,100]
ps_0 = 0.9 #It can also be chosen as random
cs_0 = 0.4 + 0j #It can also be chosen as random
for t in tau:
    print(t)
    U_tot = U_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S,t)
    rho_S = rho_S_calculation(ps_0,cs_0)
    ps_aux = []
    cs_aux = []
    for i in range(n):
        ps_aux.append(np.real(rho_S[0,0]))
        cs_aux.append(np.sqrt(np.real(rho_S[0,1])**2 + np.imag(rho_S[0,1])**2))
        new_state = U_tot@np.kron(rho_S,rho_A)@U_tot.conj().T
        rho_S = partial_trace(new_state, (2, 2), 1)
    ps.append(ps_aux)
    cs.append(cs_aux)
# %%Evolution of $p_s^{(n)}$ for different values of $\tau$
p_inf = p_A + (1-2*p_A)*(Jxx-Jyy)**2/2/(Jxx**2 + Jyy**2)
numbers = np.arange(1, n + 1)
fig, ax = plt.subplots()

numbers = np.arange(1, n + 1)

fig.suptitle(r'Evolution of $p_s^{(n)}$ for different values of $\tau$')

ax.set_title(
    rf'$\omega_S$={omega_S}, $\omega_A$={omega_A}, '
    rf'$J_x$={Jxx}, $J_y$={Jyy}, $J_z$={Jzz}, $\beta$={beta}', fontsize=10
)

ax.set_xscale('log')

for t in range(len(tau)):
    ax.plot(numbers, ps[t], label=rf'$\tau$={tau[t]}')

ax.legend()
ax.xaxis.set_label_text('Number of iterations')
ax.yaxis.set_label_text(r'$p_s^{(n)}$')
ax.set_xlim(1,1000000)
ax.set_ylim(0,1)
plt.show()
#%%Evolution of $|c_s^{(n)}|$ for different values of $\tau$
fig, ax = plt.subplots()
fig.suptitle(r'Evolution of $|c_s^{(n)}|$ for different values of $\tau$')

ax.set_title(
    rf'$\omega_S$={omega_S}, $\omega_A$={omega_A}, '
    rf'$J_x$={Jxx}, $J_y$={Jyy}, $J_z$={Jzz}, $\beta$={beta}', fontsize=10
)

ax.set_xscale('log')

for t in range(len(tau)):
    ax.plot(numbers, cs[t], label=rf'$\tau$={tau[t]}')

ax.legend()
ax.xaxis.set_label_text('Number of iterations')
ax.yaxis.set_label_text(r'$|c_s^{(n)}|$')
ax.set_xlim(1,1000000)
plt.show()



# %%
