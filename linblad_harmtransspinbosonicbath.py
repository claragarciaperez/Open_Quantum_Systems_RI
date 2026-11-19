#%%
'''
Code for obtaining the solucion of lindblad equation for:

    H = omega/2 sigma_z + omega_h a^dagger a + lambda sigma_x (a + a^dagger)

the spin is coupled to a bosonic bath with the following lindblad operators:
    L1 = sqrt(Gamma * n_B) sigma^+
    L2 = sqrt(Gamma * (n_B + 1)) sigma^-

where n_B is the number of bosons in the bath.
The expected values of sigma_x, sigma_y, sigma_z are obtained from t = 0 to t= t_max.


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
    rho = rho.reshape((2 * truncation, 2 * truncation))
    V1 = np.kron(sigma_plus(), np.eye(truncation))*np.sqrt(n_B)
    V2 = np.kron(sigma_minus(), np.eye(truncation))*np.sqrt(n_B+1)
    linblad = -1j*conmuntator(H,rho) + Gamma*(V1@rho@(V1.conj().T) - 1/2*anti_conmuntator((V1.conj().T)@V1,rho)) +  Gamma*(V2@rho@(V2.conj().T) - 1/2*anti_conmuntator((V2.conj().T)@V2,rho))
    return linblad.flatten()

#%%
class Exact_solution_single_oscillator:
    '''
    Class to obtain the exact solution of linblad equation given a single oscillator with H = omega/2 sigma_z and L = sigma^-.
    The expected values of sigma_x, sigma_y, sigma_z are obtained from t = 0 to t= t_max.

    Parameters
    ----------
    omega: float
        Frequency of the systems
    Gamma: float
        Parameter of Linclad equation (strength of the coupling with the system)
    theta: float, optional
        Angle initial state. Default value np.pi/3
    phi: float, optional
        Angle initial state. Default value np.pi/3
    t_max: float, optional
        The solution will be given from t= 0 to t= tmax. Default value 30.
    n_points: int, optional
        Number of points that will be extracted from the solution. Default value 1000.
    trincation: int, optional
        The truncation of the harmonic oscillator. Default value 4.

    '''
    def __init__(self, omega, Gamma, omega_A, omega_h,T, l,theta = np.pi/3, phi = np.pi/3, t_max =30,n_points = 10000,  truncation = 4):
        self.omega = omega
        self.Gamma = Gamma
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.n_points = n_points
        if (T==0):
            self.n_B = 0
        else:
            self.n_B = 1/(np.exp(omega_A/T)-1)
        self.T = T
        self.omega_A = omega_A
        self.omega_h = omega_h
        self.l = l
        self.truncation = truncation


    def solve(self):
        H = np.kron(sigma_z(), np.eye(self.truncation)) * self.omega / 2 + np.kron(I(),a_daggera(self.truncation))*self.omega_h + self.l*np.kron(sigma_x(),(a(self.truncation) + a_dagger(self.truncation)))
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0_spin = np.outer(state_0, state_0.conj().T)
        rho_0_harm = np.zeros((self.truncation, self.truncation), dtype=complex)
        rho_0_harm[0, 0] = 1
        rho_0 = np.kron(rho_0_spin, rho_0_harm)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma, self.n_B, self.truncation), t_eval=np.linspace(0,self.t_max,self.n_points))
    
        return solutions
    
    def expected_values(self):
        solutions = self.solve()
        dim = 2 * self.truncation
        rho_t = solutions.y.T.reshape(
            len(solutions.t), dim, dim
        )
        time = solutions.t
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(time)):
            exp_sigmaz.append(expected_sigma_z(rho_t[i], self.truncation))
            exp_sigmay.append(expected_sigma_y(rho_t[i], self.truncation))
            exp_sigmax.append(expected_sigma_x(rho_t[i], self.truncation))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]

    def save_file(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values()

        data = np.column_stack((np.real(time_cont), np.real(exp_sigmax_cont), np.real(exp_sigmay_cont),np.real(exp_sigmaz_cont)))

        np.savetxt(
        f"linblad_spinharmtrans_omega_{self.omega:.2f}_omegah_{self.omega_h:.2f}_lambda_{self.l:.2f}_Gamma_{self.Gamma:.2f}_T_{self.T:.2f}.txt",
        data,
        header="t sx sy sz",
        fmt="%.10f"
            )
    
    def plot_expectedval(self):
        time, exp_sigmax, exp_sigmay,exp_sigmaz = self.expected_values()
        plt.title(rf'Linblad Equation for a single oscillator. T = {self.T}')
        plt.plot(time, exp_sigmax, label=rf'$\langle \sigma_x\rangle$')
        plt.plot(time, exp_sigmay, label=rf'$\langle \sigma_y\rangle$')
        plt.plot(time, exp_sigmaz, label=rf'$\langle \sigma_z\rangle$')
        plt.xlabel('Time')
        plt.ylabel('Expected Value')
        plt.legend()
        plt.show()


    

    def plot_expectedval_truncation(self):

        temperatures = [0, 0.1, 0.5, 1]
        lambdas = [0, 0.1, 0.5, 1]
        truncations = [4, 6, 8, 10]

        colors = {
            'x': 'tab:blue',
            'y': 'tab:orange',
            'z': 'tab:green'
        }

        alphas = {
            4: 1.0,
            6: 0.75,
            8: 0.5,
            10: 0.25
        }

        for T in temperatures:

            self.T = T

            if T != 0:
                self.n_B = 1/(np.exp(self.omega_A/T)-1)

            for l in lambdas:

                self.l = l

                plt.figure(figsize=(10, 6))

                for truncation in truncations:

                    self.truncation = truncation

                    time, exp_sigmax, exp_sigmay, exp_sigmaz = self.expected_values()

                    alpha = alphas[truncation]

                    plt.plot(time, exp_sigmax, color=colors['x'], alpha=alpha)
                    plt.plot(time, exp_sigmay, color=colors['y'], alpha=alpha)
                    plt.plot(time, exp_sigmaz, color=colors['z'], alpha=alpha)

                component_legend = [
                    Line2D([0], [0], color=colors['x'], lw=2, label=r'$\sigma_x$'),
                    Line2D([0], [0], color=colors['y'], lw=2, label=r'$\sigma_y$'),
                    Line2D([0], [0], color=colors['z'], lw=2, label=r'$\sigma_z$')
                ]

                truncation_legend = [
                    Line2D([0], [0], color='black', alpha=alphas[t],
                        lw=2, label=f'Truncation = {t}')
                    for t in truncations
                ]

                legend1 = plt.legend(
                    handles=component_legend,
                    title='Components',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1)
                )

                plt.gca().add_artist(legend1)

                plt.legend(
                    handles=truncation_legend,
                    title='Truncation',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 0.65)
                )

                plt.title(rf'Lindblad Equation - T = {T} - $\lambda$ = {l}')
                plt.xlabel('Time')
                plt.ylabel('Expected Value')

                plt.subplots_adjust(right=0.75)

                plt.show()


        
# %%
for l in [0,0.1,0.2,0.5,1]:
    for T in [1]:
        
        Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,omega_A=0.5,omega_h=0.25,l= l,T =T, t_max=25, truncation=4).plot_expectedval()
        Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,omega_A=0.5,omega_h=0.25,l= l,T =T, t_max=25, truncation=10).plot_expectedval()

        #Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,omega_A=0.5,omega_h=0.25,l= l,T =T, t_max=25).save_file()

# %%
Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,omega_A=0.5,omega_h=0.5,l= l,T =0, t_max=25, truncation=10).plot_expectedval_truncation()

# %%
