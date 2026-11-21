#%%
'''
Code for obtaining the solucion of lindblad equation for:

    H =omega_h a^dagger a + ws/2 sigmaz + lambda sigma_x(a+a^dagger)

the spin is coupled to a bosonic bath with the following lindblad operators:
    L1 = sqrt(Gamma * n_B) a^{\dagger}
    L2 = sqrt(Gamma * (n_B + 1)) a

where n_B is the number of bosons in the bath.
The expected value of a^{\dagger}a is obtained.
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from matplotlib.lines import Line2D

#%%
def sigma_x():
    return np.array([[0,1],[1,0]])
def sigma_y():
    #return np.array([[0,-1j],[1j,0]])On the paper they use another convention for the signs
    return -1j*np.outer([0,1],[[1],[0]]) + 1j*np.outer([1,0],[[0],[1]]) 
def sigma_z():
    #return np.array([[1,0],[0,-1]]) On the paper they use another convention for the signs
    return np.outer([0,1],[[0],[1]]) - np.outer([1,0],[[1],[0]]) 

def sigma_minus():
    return np.outer([1,0],[[0], [1]])
def sigma_plus():
    return np.outer([0,1],[[1], [0]])

def I():
    return np.eye(2, dtype=complex)

def conmuntator(A,B):
    return A@B - B@A
def anti_conmuntator(A,B):
    return A@B + B@A

def expected_sigma_z(rho0,truncation=4):
    return np.trace(rho0@np.kron(sigma_z(), np.eye(truncation)))
def expected_sigma_y(rho0,truncation=4):
    return np.trace(rho0@np.kron(sigma_y(), np.eye(truncation)))
def expected_sigma_x(rho0,truncation=4):
    return np.trace(rho0@np.kron(sigma_x(), np.eye(truncation)))

def expected_a_daggera(rho0,truncation=4):
    return np.trace(rho0@(np.kron(I(),a_daggera(truncation))))
def RZ(alpha):
    Z = -sigma_z()
    return(expm(-1j * alpha / 2 *Z))

def RY(alpha):
    Y = -sigma_y()
    return(expm(-1j * alpha / 2 *Y))

def partial_trace(rho, dims, subsystem):
    dA, dB = dims
    rho = rho.reshape(dA, dB, dA, dB)

    if subsystem == 0:  # trazar A
        return np.trace(rho, axis1=0, axis2=2)
    elif subsystem == 1:  # trazar B
        return np.trace(rho, axis1=1, axis2=3)

def a(truncation = 4):
    a = np.zeros((truncation, truncation), dtype=complex)
    for i in range(1, truncation):
        a[i-1, i] = np.sqrt(i)
    return a

def a_dagger(truncation = 4):   
    a_dagger = np.zeros((truncation, truncation), dtype=complex)
    for i in range(1, truncation):
        a_dagger[i, i-1] = np.sqrt(i)
    return a_dagger

def a_daggera(truncation = 4):
    return a_dagger(truncation)@a(truncation)

def Linblad(t, rho, H,Gamma,n_B,truncation = 4):
    rho = rho.reshape((2*truncation,  2*truncation))
    V1 = np.kron(I(),a_dagger(truncation))*np.sqrt(n_B)
    V2 = np.kron(I(),a(truncation))*np.sqrt(n_B+1)
    linblad = -1j*conmuntator(H,rho) + Gamma*(V1@rho@(V1.conj().T) - 1/2*anti_conmuntator((V1.conj().T)@V1,rho)) +  Gamma*(V2@rho@(V2.conj().T) - 1/2*anti_conmuntator((V2.conj().T)@V2,rho))
    return linblad.flatten()

#%%
class Exact_solution_spin_harmoscil_bath:
    '''
    Class to obtain the exact solution of linblad equation given a single oscillator with H =omega_h a^dagger a + ws/2 sigmaz + lambda sigma_x(a+a^dagger), and
        L1 = sqrt(Gamma * n_B) a^{\dagger}
        L2 = sqrt(Gamma * (n_B + 1)) a
    The expected values of n, sigma_z are obtained from t = 0 to t= t_max.

    Parameters
    ----------
    Gamma: float
        Parameter of Linclad equation (strength of the coupling with the system)
    omega: float
        Frequency of the spin
    omega_h: float
        Frequency of the harmonic oscillator
    omega_A: float
        Frequency of the bath
    l: float
        Parameter lambda of the strength of the interaction spin-harmonic oscillator
    T: float
        Temperature of the bath
    theta: float, optional
        Angle initial state. Default value np.pi/3
    phi: float, optional
        Angle initial state. Default value np.pi/3
    t_max: float, optional
        The solution will be given from t= 0 to t= tmax. Default value 30.
    n_points: int, optional
        Number of points that will be extracted from the solution. Default value 1000.
    truncation: int, optional
        The truncation of the harmonic oscillator. Default value 4.

    '''
    def __init__(self, Gamma,omega, omega_h,omega_A,l,T, theta = np.pi/3, phi = np.pi/3, t_max =30,n_points = 10000,  truncation = 4):
        self.Gamma = Gamma
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.n_points = n_points
        self.omega_A = omega_A
        if (T==0):
            self.n_B = 0
        else:
            self.n_B = 1/(np.exp(omega_A/T)-1)
        self.T = T
        self.omega_h = omega_h
        self.truncation = truncation
        self.omega = omega
        self.l = l


    def solve(self):
        H = np.kron(sigma_z(), np.eye(self.truncation)) * self.omega / 2 + np.kron(I(),a_daggera(self.truncation))*self.omega_h + self.l*np.kron(sigma_x(),(a(self.truncation) + a_dagger(self.truncation)))
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0= np.zeros((self.truncation, self.truncation), dtype=complex)
        rho_0_spin = np.outer(state_0, state_0.conj().T)
        rho_0_harm = np.zeros((self.truncation, self.truncation), dtype=complex)
        rho_0_harm[0, 0] = 1
        rho_0 = np.kron(rho_0_spin, rho_0_harm)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma, self.n_B, self.truncation), t_eval=np.linspace(0,self.t_max,self.n_points))
    
        return solutions
    
    def expected_values(self):
        solutions = self.solve()
        dim = 2*self.truncation
        rho_t = solutions.y.T.reshape(
            len(solutions.t), dim, dim
        )
        time = solutions.t
        exp_a_daggera = []
        sz_spin = []
        for i in range(len(time)):
            exp_a_daggera.append(expected_a_daggera(rho_t[i], self.truncation))
            sz_spin.append(expected_sigma_z(rho_t[i],self.truncation))

        return [time,exp_a_daggera,sz_spin]

    def save_file(self):
        time_cont, n,sz_spin = self.expected_values()

        data = np.column_stack((np.real(time_cont), np.real(n), np.real(sz_spin)))

        np.savetxt(
        f"linblad_spinharmosctran_omega_{self.omega:.2f}_omegah_{self.omega_h:.2f}_omegaA_{self.omega_A:.2f}_l_{self.l:.2f}_Gamma_{self.Gamma:.2f}_T_{self.T:.2f}.txt",
        data,
        header="t adaggera sz_spin",
        fmt="%.10f"
            )
    
    def plot_expectedval(self):
        time, adaggera,sz= self.expected_values()
        plt.title(rf'Linblad Equation for a harmonic oscillator. T = {self.T}')
        plt.plot(time, adaggera)
        plt.xlabel('Time')
        plt.ylabel(r'$\langle a a^{\dagger}\rangle$')
        plt.legend()
        plt.show()
        plt.title(rf'Linblad Equation for a harmonic oscillator. T = {self.T}')
        plt.plot(time, sz)
        plt.xlabel('Time')
        plt.ylabel(r'$\langle\sigma_z\rangle$')
        plt.legend()
        plt.show()
        


    def plot_expectedval_temperature(self):
        fig_x, ax_x = plt.subplots()
        fig_y, ax_y = plt.subplots()
        for T in [0, 0.1, 0.25, 0.5, 1]:
            self.T = T

            if T == 0:
                self.n_B = 0
            else:
                self.n_B = 1/(np.exp(self.omega_A/T)-1)

            time, adaggera, sz = self.expected_values()

            # Plot de <a†a>
            ax_x.plot(time, adaggera, label=rf'T = {T:.2f}')

            # Plot de <σz>
            ax_y.plot(time, sz, label=rf'T = {T:.2f}')


        # Primera figura: <a†a>
        ax_x.set_title(rf'Linblad Equation - $\lambda = ${self.l:.2f}')
        ax_x.set_xlabel('Time')
        ax_x.set_ylabel(r'$\langle a^\dagger a\rangle$')
        ax_x.legend()

        # Segunda figura: <σz>
        ax_y.set_title(rf'Linblad Equation - $\lambda = ${self.l:.2f}')
        ax_y.set_xlabel('Time')
        ax_y.set_ylabel(r'$\langle \sigma_z\rangle$')
        ax_y.legend()

        plt.show()
    

    
# %%
for T in [0,0.1,0.25,0.5,1]:
    Exact_solution_spin_harmoscil_bath(omega = 0.5, Gamma = 0.25,omega_A=0.25,omega_h=0.5,l= 0.25,T =T, t_max=25, truncation=4).save_file()

    Exact_solution_spin_harmoscil_bath(omega = 0.5, Gamma = 0.25,omega_A=0.5,omega_h=0.5,l= 0.25,T =T, t_max=25, truncation=4).plot_expectedval()



