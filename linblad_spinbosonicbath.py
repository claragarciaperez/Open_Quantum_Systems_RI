#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
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

def expected_sigma_z(rho00,rho01,rho10,rho11):
    rho = np.array([[rho00,rho01],[rho10,rho11]])
    return np.trace(rho@sigma_z())
def expected_sigma_y(rho00,rho01,rho10,rho11):
    rho = np.array([[rho00,rho01],[rho10,rho11]])
    return np.trace(rho@sigma_y())
def expected_sigma_x(rho00,rho01,rho10,rho11):
    rho = np.array([[rho00,rho01],[rho10,rho11]])
    return np.trace(rho@sigma_x())

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



def Linblad(t, rho, H,Gamma,n_B):
    rho = rho.reshape((2, 2))
    V1 = sigma_plus()*np.sqrt(n_B)
    V2 = sigma_minus()*np.sqrt(n_B+1)
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

    '''
    def __init__(self, omega, Gamma, omega_A, T,delta_t, theta = np.pi/3, phi = np.pi/3, t_max =30,n_points = 1000, trott = True):
        self.omega = omega
        self.Gamma = Gamma
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.n_points = n_points
        self.n_B = 1/(np.exp(omega_A/T)-1)
        self.T = T
        self.g  = np.sqrt(Gamma*(2*self.n_B+1)/delta_t)
        self.delta_t = delta_t
        self.trott = trott
        self.omega_A = omega_A


    def solve(self):
        print(self.omega)
        H = sigma_z() * self.omega / 2
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma, self.n_B), t_eval=np.linspace(0,self.t_max,self.n_points))
        return solutions
    
    def expected_values(self):
        solutions = self.solve()
        rho_00 = solutions.y[0]
        rho_01 = solutions.y[1]
        rho_10 = solutions.y[2]
        rho_11 = solutions.y[3]
        time = solutions.t
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(time)):
            exp_sigmaz.append(expected_sigma_z(rho_00[i],rho_01[i],rho_10[i],rho_11[i]))
            exp_sigmay.append(expected_sigma_y(rho_00[i],rho_01[i],rho_10[i],rho_11[i]))
            exp_sigmax.append(expected_sigma_x(rho_00[i],rho_01[i],rho_10[i],rho_11[i]))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]

    def save_file(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values()

        data = np.column_stack((np.real(time_cont), np.real(exp_sigmax_cont), np.real(exp_sigmay_cont),np.real(exp_sigmaz_cont)))

        np.savetxt(
        f"cont_solution_omega_{self.omega}_Gamma_{self.Gamma}.txt",
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

    def V(self):
            return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
        
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        if self.trott == True:
            H_ancilla = np.kron(I(),self.omega_A/2*sigma_z())
            sol =(expm(-1j * H_ancilla *self.delta_t))@(expm(-1j * H *self.delta_t))@(expm(-1j * self.V() *self.delta_t))
        else: 
            sol = expm(-1j * (H + self.V()) *self.delta_t)
        return sol
    
    def collision_map(self,H,rho):
        #pa = 1-np.exp(-self.omega_A / self.T) / (1.0 + np.exp(-self.omega_A / self.T))
        #pa = 1/(1+np.exp(-self.omega_A/self.T))
        #pa = np.exp(-self.n_B/ self.T) / (1.0 + np.exp(-self.n_B/ self.T))
        q = np.exp(-self.omega_A / self.T)
        pa = 1.0 / (1.0 + q)
        thermal_state = np.array([
        [pa, 0],
        [0, 1-pa]
    ])
        state = np.kron(rho,thermal_state)
        final_state =  self.U(H)@state@(self.U(H).conj().T)
        return partial_trace(final_state, (2,2),1)

    def solve_RI(self):
           state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
           rho_0 = np.outer(state_0, state_0.conj().T)
           H = sigma_z() *self.omega / 2
           niter = int(self.t_max/self.delta_t)+1
           rho = [rho_0]
           for i in range(1,niter):
               rho.append(self.collision_map(H,rho[i-1]))
           return rho
           
    def expected_values_RI(self):
        rho = self.solve_RI()
        time = np.arange(len(rho)) * self.delta_t
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(rho)):
            exp_sigmax.append(np.trace(rho[i]@sigma_x()))
            exp_sigmay.append(np.trace(rho[i]@sigma_y()))
            exp_sigmaz.append(np.trace(rho[i]@sigma_z()))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]
    def plot_expectedval_total(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values()
        x_cont = time_cont
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values_RI()
        x_col = time_col
        plt.plot(x_cont, exp_sigmax_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
        plt.plot(x_cont, exp_sigmay_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
        plt.plot(x_cont, exp_sigmaz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')
        plt.scatter(x_col, exp_sigmax_col,
            facecolors='none', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ RI trotter')

        plt.scatter(x_col, exp_sigmay_col,
                    facecolors='none', edgecolors='orange',
                    label=rf'$\langle \sigma_y\rangle$ RI trotter')

        plt.scatter(x_col, exp_sigmaz_col,
                    facecolors='none', edgecolors='green',
                    label=rf'$\langle \sigma_z\rangle$ RI trotter')
        plt.xlabel('gt')
        plt.title(rf'T = {self.T}')
        plt.ylabel('Expected Value')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
    def save_file_cont(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values()

        data = np.column_stack((np.real(time_cont), np.real(exp_sigmax_cont), np.real(exp_sigmay_cont),np.real(exp_sigmaz_cont)))

        np.savetxt(
        f"linblad_Gamma_{self.Gamma}_omegas_{self.omega}_omegaA_{self.omega_A}_T_{self.T:.2f}.txt",
        data,
        header="t sx sy sz",
        fmt="%.10f"
            )
# %%
for T in [0.01,0.1,0.5,1,10]:
    Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,omega_A=0.5,T = T,delta_t=0.1 ).plot_expectedval_total()
    Exact_solution_single_oscillator(omega = 0.5, Gamma = 0.25,delta_t=0.01, omega_A=0.5, T= T,t_max=25).save_file_cont()

# %%
