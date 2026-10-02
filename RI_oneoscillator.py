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
omega_S = 1
omega_A = 2
Jxx = 2
Jyy = 1
Jzz = 0
#tau = 0.01
beta = 1
n = 1000000
#%%
H_tot =H_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S)
rho_A = np.array([[1/(1+np.exp(-beta*omega_A)),0],[0,1/(1+np.exp(beta*omega_A))]])
#%% Calculation of ps evolution for different values of tau
ps = []
tau = [0.001,0.01,0.1,1,10,100]
cs = 0.4
for t in tau:
    print(t)
    U_tot = U_tot_calculation(Jxx,Jyy,Jzz,omega_A,omega_S,t)
    rho_S = rho_S_calculation(0.9 + 0j,cs)
    ps_aux = []
    for i in range(n):
        ps_aux.append(np.real(rho_S[0,0]))
        new_state = U_tot@np.kron(rho_S,rho_A)@U_tot.conj().T
        rho_S = partial_trace(new_state, (2, 2), 1)
    ps.append(ps_aux)
# %%
numbers = np.arange(1, n + 1)
plt.xscale('log')
for t in range(len(tau)):
    plt.plot(numbers, ps[t], label=rf'$\tau$={tau[t]}')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel(r'$p_s^{(n)}$')

# %%
