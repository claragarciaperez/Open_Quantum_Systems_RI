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

def L():
        return sigma_minus()

def Linblad(t, rho, H,Gamma):
    rho = rho.reshape((2, 2))
    linblad = -1j*conmuntator(H,rho) + Gamma*(L()@rho@(L().conj().T) - 1/2*anti_conmuntator((L().conj().T)@L(),rho))
    return linblad.flatten()
#%%
class Exact_solution_single_oscillator:
    def __init__(self, omega, Gamma, theta = np.pi/3, phi = np.pi/3, t_max =30,n_points = 1000 ):
        self.omega = omega
        self.Gamma = Gamma
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.n_points = n_points


    def solve(self):
        print(self.omega)
        H = sigma_z() * self.omega / 2
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma), t_eval=np.linspace(0,self.t_max,self.n_points))
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
    
    def plot_expectedval(self):
        time, exp_sigmax, exp_sigmay,exp_sigmaz = self.expected_values()
        plt.title('Linblad Equation for a single oscillator')
        plt.plot(time, exp_sigmax, label=rf'$\langle \sigma_x\rangle$')
        plt.plot(time, exp_sigmay, label=rf'$\langle \sigma_y\rangle$')
        plt.plot(time, exp_sigmaz, label=rf'$\langle \sigma_z\rangle$')
        plt.xlabel('Time')
        plt.ylabel('Expected Value')
        plt.legend()
        plt.show()


class RI_single_oscillator:
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, t_max =40 ):
            self.g = g
            self.delta_t = delta_t
            self.theta = theta
            self.phi = phi
            self.t_max = t_max
    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        return (expm(-1j * H *self.delta_t))@(expm(-1j * self.V() *self.delta_t))

    def collision_map(self,H,rho):
        state = np.kron(rho,np.outer([1,0],[[1],[0]]))
        final_state =  self.U(H)@state@(self.U(H).conj().T)
        return partial_trace(final_state, (2,2),1)

    def solve(self):
        omega = 2*self.g
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        H = sigma_z() *omega / 2
        niter = int(self.t_max/self.delta_t)
        rho = [rho_0]
        for i in range(1,niter):
            rho.append(self.collision_map(H,rho[i-1]))
        return rho
    
    def expected_values(self):
            rho = self.solve()
            niter = int(self.t_max/self.delta_t)
            time = np.linspace(0,self.t_max,niter)
            exp_sigmaz = []
            exp_sigmay = []
            exp_sigmax = []
            for i in range(len(rho)):
                exp_sigmax.append(np.trace(rho[i]@sigma_x()))
                exp_sigmay.append(np.trace(rho[i]@sigma_y()))
                exp_sigmaz.append(np.trace(rho[i]@sigma_z()))
            return [time,exp_sigmax,exp_sigmay,exp_sigmaz]
    
    def plot_expectedval(self):
        time, exp_sigmax, exp_sigmay,exp_sigmaz = self.expected_values()
        plt.title('Collision Model with Trottersization for a single oscillator')
        plt.plot(time, exp_sigmax, label=rf'$\langle \sigma_x\rangle$')
        plt.plot(time, exp_sigmay, label=rf'$\langle \sigma_y\rangle$')
        plt.plot(time, exp_sigmaz, label=rf'$\langle \sigma_z\rangle$')
        plt.xlabel('Time')
        plt.ylabel('Expected Value')
        plt.legend()
        plt.show()


class continuous_trotter_collision:
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, t_max =30, npoints_cont = 1000 ):
        self.g = g
        self.delta_t = delta_t
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.omega = 2*g
        self.Gamma = g/2
        self.npoints_cont = npoints_cont

    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        return (expm(-1j * H *self.delta_t))@(expm(-1j * self.V() *self.delta_t))

    def collision_map(self,H,rho):
        state = np.kron(rho,np.outer([1,0],[[1],[0]]))
        final_state =  self.U(H)@state@(self.U(H).conj().T)
        return partial_trace(final_state, (2,2),1)

    def solve_cont(self):
        print(self.omega)
        H = sigma_z() * self.omega / 2
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma), t_eval=np.linspace(0,self.t_max,self.npoints_cont))
        return solutions

    def expected_values_cont(self):
        solutions = self.solve_cont()
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

    def solve_RI(self):
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        H = sigma_z() *self.omega / 2
        niter = int(self.t_max/self.delta_t)
        rho = [rho_0]
        for i in range(1,niter):
            rho.append(self.collision_map(H,rho[i-1]))
        return rho
        
    def expected_values_RI(self):
        rho = self.solve_RI()
        niter = int(self.t_max/self.delta_t)
        time = np.linspace(0,self.t_max,niter)
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(rho)):
            exp_sigmax.append(np.trace(rho[i]@sigma_x()))
            exp_sigmay.append(np.trace(rho[i]@sigma_y()))
            exp_sigmaz.append(np.trace(rho[i]@sigma_z()))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]

    def plot_expectedval(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values_cont()
        x_cont = time_cont*self.g
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values_RI()
        x_col = time_col*self.g
        plt.plot(x_cont, exp_sigmax_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
        plt.plot(x_cont, exp_sigmay_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
        plt.plot(x_cont, exp_sigmaz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')
        plt.scatter(x_col, exp_sigmax_col, label=rf'$\langle \sigma_x\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmay_col, label=rf'$\langle \sigma_y\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmaz_col, label=rf'$\langle \sigma_z\rangle$ RI trotter')
        plt.xlabel('gt')
        plt.ylabel('Expected Value')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()


class continuous_collision_qiskit:
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, t_max =30, npoints_cont = 1000 , file = "qiskit2_sxsysz.txt"):
        self.g = g
        self.delta_t = delta_t
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.omega = 2*g
        self.Gamma = g/2
        self.npoints_cont = npoints_cont
        self.file = file

    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        return (expm(-1j * H *self.delta_t))@(expm(-1j * self.V() *self.delta_t))

    def collision_map(self,H,rho):
        state = np.kron(rho,np.outer([1,0],[[1],[0]]))
        final_state =  self.U(H)@state@(self.U(H).conj().T)
        return partial_trace(final_state, (2,2),1)

    def solve_cont(self):
        print(self.omega)
        H = sigma_z() * self.omega / 2
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        solutions = solve_ivp(Linblad,[0,self.t_max], rho_0.flatten(), args=(H, self.Gamma), t_eval=np.linspace(0,self.t_max,self.npoints_cont))
        return solutions

    def expected_values_cont(self):
        solutions = self.solve_cont()
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

    def solve_RI(self):
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        H = sigma_z() *self.omega / 2
        niter = int(self.t_max/self.delta_t)
        rho = [rho_0]
        for i in range(1,niter):
            rho.append(self.collision_map(H,rho[i-1]))
        return rho
        
    def expected_values_RI(self):
        rho = self.solve_RI()
        niter = int(self.t_max/self.delta_t)
        time = np.linspace(0,self.t_max,niter)
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(rho)):
            exp_sigmax.append(np.trace(rho[i]@sigma_x()))
            exp_sigmay.append(np.trace(rho[i]@sigma_y()))
            exp_sigmaz.append(np.trace(rho[i]@sigma_z()))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]

    def plot_expectedval(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values_cont()
        x_cont = time_cont*self.g
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values_RI()
        x_col = time_col*self.g
        data_qiskit = np.loadtxt(self.file)
        x_qiskit  = data_qiskit[:, 0]*self.g
        sx_qiskit= data_qiskit[:, 1]
        sy_qiskit = data_qiskit[:, 2]
        sz_qiskit = data_qiskit[:, 3]
        plt.plot(x_cont, exp_sigmax_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
        plt.plot(x_cont, exp_sigmay_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
        plt.plot(x_cont, exp_sigmaz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')
        plt.scatter(x_col, exp_sigmax_col, label=rf'$\langle \sigma_x\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmay_col, label=rf'$\langle \sigma_y\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmaz_col, label=rf'$\langle \sigma_z\rangle$ RI trotter')
        plt.scatter(
            x_qiskit, sx_qiskit,
            marker='^', facecolors='none', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ Qiskit (No noise)'
        )

        plt.scatter(
            x_qiskit, sy_qiskit,
            marker='^', facecolors='none', edgecolors='orange',
            label=rf'$\langle \sigma_y\rangle$ Qiskit (No noise)'
        )

        plt.scatter(
            x_qiskit, sz_qiskit,
            marker='^', facecolors='none', edgecolors='green',
            label=rf'$\langle \sigma_z\rangle$ Qiskit (No noise)'
        )
        plt.xlabel('gt')
        plt.ylabel('Expected Value')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
    
#%%
g = 0.5
delta_t = 1
continuous_trotter_collision(g,delta_t).plot_expectedval()
continuous_collision_qiskit(g,delta_t).plot_expectedval()

# %%
