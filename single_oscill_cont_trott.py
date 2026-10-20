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
        plt.title('Linblad Equation for a single oscillator')
        plt.plot(time, exp_sigmax, label=rf'$\langle \sigma_x\rangle$')
        plt.plot(time, exp_sigmay, label=rf'$\langle \sigma_y\rangle$')
        plt.plot(time, exp_sigmaz, label=rf'$\langle \sigma_z\rangle$')
        plt.xlabel('Time')
        plt.ylabel('Expected Value')
        plt.legend()
        plt.show()


class RI_single_oscillator_trott:
    '''
    Class to obtain the  solution of linblad equation given a single oscillator with H = omega/2 sigma_z under repeated interaction (or collision model) protocol
    where trotterization is implemented. It gives the expected values for sigma_x, sigma_y, sigma_z from t = 0 to t= t_max, in temporal steps delta_t.

    Parameters
    ----------
    g: float
        Interaction parameter. (omega = 2g, Gamma = g^2 delta t)
    delta_t: float
        Size of the temporal step
    theta: float, optional
        Angle initial state. Default value np.pi/3
    phi: float, optional
        Angle initial state. Default value np.pi/3
    t_max: float, optional
        The solution will be given from t= 0 to t= tmax. Default value 30.

    '''
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, omega = 1, t_max =30 , trott = True):
        self.g = g
        self.delta_t = delta_t
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.trott = trott
        self.omega = omega
    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        if self.trott == True:
            sol =(expm(-1j * H *self.delta_t))@(expm(-1j * self.V() *self.delta_t))
        else: 
            sol = expm(-1j * (H + self.V()) *self.delta_t)
        return sol
    
    def collision_map(self,H,rho):
        state = np.kron(rho,np.outer([1,0],[[1],[0]]))
        final_state =  self.U(H)@state@(self.U(H).conj().T)
        return partial_trace(final_state, (2,2),1)

    def solve(self):
        omega = self.omega
        state_0 = RZ(-self.phi)@RY(np.pi-self.theta)@ np.array([1, 0], dtype=complex)
        rho_0 = np.outer(state_0, state_0.conj().T)
        H = sigma_z() *omega / 2
        niter = int(self.t_max/self.delta_t) + 1
        rho = [rho_0]
        for i in range(1,niter):
            rho.append(self.collision_map(H,rho[i-1]))
        return rho
    
    def expected_values(self):
        rho = self.solve()
        time = np.arange(len(rho)) * self.delta_t
        exp_sigmaz = []
        exp_sigmay = []
        exp_sigmax = []
        for i in range(len(rho)):
            exp_sigmax.append(np.trace(rho[i]@sigma_x()))
            exp_sigmay.append(np.trace(rho[i]@sigma_y()))
            exp_sigmaz.append(np.trace(rho[i]@sigma_z()))
        return [time,exp_sigmax,exp_sigmay,exp_sigmaz]

    def save_file_RI(self):
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values()

        data = np.column_stack((np.real(time_col),  np.real(exp_sigmax_col), np.real(exp_sigmay_col),np.real(exp_sigmaz_col)))

        if self.trott == True:
            np.savetxt(
            f"RI_solution_trott_dt_{self.delta_t}_g_{self.g}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )
        else:
            np.savetxt(
            f"RI_solution_notrott_dt_{self.delta_t}_g_{self.g}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )
    
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


class continuous_collision:
    '''
    Class to obtain the exact solution of linblad equation given a single oscillator with H = omega/2 sigma_z and L = sigma^- as well as 
    under repeated interaction (or collision model) protocol where trotterization is implemented. 
    It gives the expected values for sigma_x, sigma_y, sigma_z from t = 0 to t= t_max, for the exat solution and from RI in temporal steps delta_t.

    Para∫meters
    ----------
    g: float
        Interaction parameter. (omega = 2g, Gamma = g^2 dt)
    delta_t: float
        Size of the temporal step
    theta: float, optional
        Angle initial state. Default value np.pi/3
    phi: float, optional
        Angle initial state. Default value np.pi/3
    t_max: float, optional
        The solution will be given from t= 0 to t= tmax. Default value 30.
    n_points_cont: int, optional
        Number of points that will be extracted from the solution in the exact model. Default value 1000.
    trott: boal, optional
        Whether the operator in RI has trotterization. Default is true
    
    '''
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, omega = 1,t_max =30, npoints_cont = 1000, trott = True ):
        self.g = g
        self.delta_t = delta_t
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.omega = omega
        self.Gamma = g**2*delta_t
        self.npoints_cont = npoints_cont
        self.trott = trott

    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        if self.trott == True:
            sol =(expm(-1j *H  *self.delta_t))@(expm(-1j * self.V() *self.delta_t))
        else: 
            Ham = (H + self.V())
            sol = expm(-1j * Ham *self.delta_t)
        return sol

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

    def save_file_cont(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values_cont()

        data = np.column_stack((np.real(time_cont), np.real(exp_sigmax_cont), np.real(exp_sigmay_cont),np.real(exp_sigmaz_cont)))

        np.savetxt(
        f"cont_solution_dt_{self.delta_t}_g_{self.g:.2f}_omegas_{self.omega}.txt",
        data,
        header="t sx sy sz",
        fmt="%.10f"
            )
    
    def save_file_RI(self):
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values_RI()

        data = np.column_stack((np.real(time_col),  np.real(exp_sigmax_col), np.real(exp_sigmay_col),np.real(exp_sigmaz_col)))

        if self.trott == True:
            np.savetxt(
            f"RI_solution_trott_dt_{self.delta_t}_g_{self.g:.2f}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )
        else:
            np.savetxt(
            f"RI_solution_notrott_dt_{self.delta_t}_g_{self.g:.2f}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )

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
    '''
    Class to obtain the exact solution of linblad equation given a single oscillator with H = omega/2 sigma_z and L = sigma^- as well as 
    under repeated interaction (or collision model) protocol where trotterization is implemented. 
    It gives the expected values for sigma_x, sigma_y, sigma_z from t = 0 to t= t_max, for the excat solution and from RI in temporal steps delta_t.
    The expected values using a quantum circuit with qiskit are also represented.

    Parameters
    ----------
    g: float
        Interaction parameter. (omega = 2g, Gamma = g^2 delta_t)
    delta_t: float
        Size of the temporal step
    theta: float, optional
        Angle initial state. Default value np.pi/3
    phi: float, optional
        Angle initial state. Default value np.pi/3
    t_max: float, optional
        The solution will be given from t= 0 to t= tmax. Default value 30.
    n_points_cont: int, optional
        Number of points that will be extracted from the solution in the exact model. Default value 1000.
    file: string, optional
        File with time and the expected value for sigma_x, sigma_y, sigma_z using qiskit. Default "qiskit2_kingston_sxsysz.txt"
    ns: int, optional
        Number of shots used on qiskit. Default is 1024
    file: string, optional
            Number of sigmas for the errors bars for data from qiskit. Default is 2
    
    '''
    def __init__(self, g, delta_t, theta = np.pi/3, phi = np.pi/3, omega = 1, t_max =30, npoints_cont = 1000 , trott = True , file = "qiskit4_fez_sxsysz_ns1024.txt", ns = 1024, sigmas = 2, errorbars = True, label_qiskit = '(no noise)'):
        self.g = g
        self.delta_t = delta_t
        self.theta = theta
        self.phi = phi
        self.t_max = t_max
        self.omega = omega
        self.Gamma = g**2*delta_t
        self.npoints_cont = npoints_cont
        self.file = file
        self.ns = ns
        self.sigmas = sigmas
        self.errorbars = errorbars
        self.label_qiskit = label_qiskit
        self.trott = trott

    def V(self):
        return self.g*(np.kron(sigma_minus(),sigma_plus()) + np.kron(sigma_plus(),sigma_minus()))
    
    def U(self,Hstate):
        H = np.kron(Hstate, I())
        if self.trott == True:
            sol =(expm(-1j *H  *self.delta_t))@(expm(-1j * self.V() *self.delta_t))
        else: 
            Ham = (H + self.V())
            sol = expm(-1j * Ham *self.delta_t)
        return sol

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

    def save_file_cont(self):
        time_cont, exp_sigmax_cont, exp_sigmay_cont,exp_sigmaz_cont = self.expected_values_cont()
        data = np.column_stack((np.real(time_cont), np.real(exp_sigmax_cont), np.real(exp_sigmay_cont),np.real(exp_sigmaz_cont)))

        np.savetxt(
        f"cont_solution_dt_{self.delta_t}_g_{self.g}.txt",
        data,
        header="t sx sy sz",
        fmt="%.10f"
            )
    
    def save_file_RI(self):
        time_col, exp_sigmax_col, exp_sigmay_col,exp_sigmaz_col = self.expected_values_RI()

        data = np.column_stack((np.real(time_col),  np.real(exp_sigmax_col), np.real(exp_sigmay_col),np.real(exp_sigmaz_col)))

        if self.trott == True:
            np.savetxt(
            f"RI_solution_trott_dt_{self.delta_t}_g_{self.g}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )
        else:
            np.savetxt(
            f"RI_solution_notrott_dt_{self.delta_t}_g_{self.g}_omegas_{self.omega}.txt",
            data,
            header="t sx sy sz",
            fmt="%.10f"
            )

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

        error_sx_qiskit = []
        for i in range(len (sx_qiskit)):
            error_sx_qiskit.append(self.sigmas*np.sqrt((1-sx_qiskit[i]**2)/self.ns ))
        error_sy_qiskit = []
        for i in range(len (sy_qiskit)):
            error_sy_qiskit.append(self.sigmas*np.sqrt((1-sy_qiskit[i]**2)/self.ns) )
        error_sz_qiskit = []
        for i in range(len (sz_qiskit)):
            error_sz_qiskit.append(self.sigmas*np.sqrt((1-sz_qiskit[i]**2)/self.ns ))


        #plt.title('IBM_kingston - Refresh Ancilla')
        plt.plot(x_cont, exp_sigmax_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
        plt.plot(x_cont, exp_sigmay_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
        plt.plot(x_cont, exp_sigmaz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')
        plt.scatter(x_col, exp_sigmax_col, label=rf'$\langle \sigma_x\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmay_col, label=rf'$\langle \sigma_y\rangle$ RI trotter')
        plt.scatter(x_col, exp_sigmaz_col, label=rf'$\langle \sigma_z\rangle$ RI trotter')
        

        if self.errorbars == True:

            plt.errorbar(
                x_qiskit,
                sz_qiskit,
                yerr=error_sz_qiskit,
                xerr=0,
                fmt='^',
                markerfacecolor='none',
                markeredgecolor='green',
                ecolor='green',
                label=rf'$\langle \sigma_z\rangle$ Qiskit {self.label_qiskit}'
            )
            plt.errorbar(
                        x_qiskit,
                        sx_qiskit,
                        yerr=error_sx_qiskit,
                        xerr=0,
                        fmt='^',
                        markerfacecolor='none',
                        markeredgecolor='blue',
                        ecolor='blue',
                        label=rf'$\langle \sigma_x\rangle$ Qiskit {self.label_qiskit}'
                    )
            plt.errorbar(
                        x_qiskit,
                        sy_qiskit,
                        yerr=error_sy_qiskit,
                        xerr=0,
                        fmt='^',
                        markerfacecolor='none',
                        markeredgecolor='orange',
                        ecolor='orange',
                        label=rf'$\langle \sigma_y\rangle$ Qiskit {self.label_qiskit}'
                    )
        else:

            plt.scatter(
                       x_qiskit, sx_qiskit,
                        marker='^', facecolors='none', edgecolors='blue',
                        label=rf'$\langle \sigma_x\rangle$ Qiskit {self.label_qiskit}'
                    )
                    
                    
            
            
            plt.scatter(
                x_qiskit, sy_qiskit,
                marker='^', facecolors='none', edgecolors='orange',
                label=rf'$\langle \sigma_y\rangle$ Qiskit {self.label_qiskit}'
            )
    
            plt.scatter(
                x_qiskit, sz_qiskit,
                  marker='^', facecolors='none', edgecolors='green',
                   label=rf'$\langle \sigma_z\rangle$ Qiskit {self.label_qiskit}'
            )

        plt.xlabel('gt')
        plt.ylabel('Expected Value')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
    
#%%
#if __name__ == "__main__":
delta_t = 1
g = 0.5/delta_t
#continuous_collision(g,delta_t, omega = 2g).plot_expectedval()
#continuous_collision_qiskit(g,delta_t, omega = 2g) file = "qiskit4_kingston_sxsysz_ns1024.txt", ns = 1024, errorbars=False, label_qiskit='IBM_kingston').plot_expectedval()
#continuous_collision_qiskit(g,delta_t, omega = 2g)).save_file_cont()
#continuous_collision_qiskit(g,delta_t, omega = 2g) trott=False).save_file_RI()
list_deltat = [0.05,0.1,0.5,0.75,1,5]
gamma = 0.25
for delta_t in list_deltat:
    g = np.sqrt(gamma/delta_t)
    #g = 0.5/delta_t
#   #g = 0.5
    continuous_collision(g,delta_t,  trott=True).save_file_RI()
    continuous_collision(g,delta_t ,trott=False).save_file_RI()
    continuous_collision(g,delta_t,trott=True).save_file_cont()

# %%
g = 0.5
data_cont = np.loadtxt("cont_solution.txt")
x_cont  = data_cont[:, 0]*g
sx_cont= data_cont[:, 1]
sy_cont = data_cont[:, 2]
sz_cont = data_cont[:, 3]

for delta_t in list_deltat:
   #g_aux = 0.5/delta_t
    g_aux = np.sqrt(gamma/delta_t)
    data_cont_deltat = np.loadtxt(f"cont_solution_dt_{delta_t}_g_{g_aux:.2f}_omegas_1.txt")

    globals()[f"data_cont_{delta_t}"] = data_cont_deltat
    globals()[f"x_cont_{delta_t}"] = data_cont_deltat[:, 0]*g_aux
    globals()[f"sx_cont_{delta_t}"] = data_cont_deltat[:, 1]
    globals()[f"sy_cont_{delta_t}"] = data_cont_deltat[:, 2]
    globals()[f"sz_cont_{delta_t}"] = data_cont_deltat[:, 3]

data_RI_trott = np.loadtxt("RI_solution_trott.txt")
x_RI_trott  = data_RI_trott[:, 0]*g
sx_RI_trott= data_RI_trott[:, 1]
sy_RI_trott = data_RI_trott[:, 2]
sz_RI_trott = data_RI_trott[:, 3]

for delta_t in list_deltat:
    #g_aux= 0.5/delta_t
    g_aux = np.sqrt(gamma/delta_t)
    data_RI_trott_deltat = np.loadtxt(f"RI_solution_trott_dt_{delta_t}_g_{g_aux:.2f}_omegas_1.txt")

    globals()[f"data_RI_trott_{delta_t}"] = data_RI_trott_deltat
    globals()[f"x_RI_trott_{delta_t}"] = data_RI_trott_deltat[:, 0]*g_aux 
    globals()[f"sx_RI_trott_{delta_t}"] = data_RI_trott_deltat[:, 1]
    globals()[f"sy_RI_trott_{delta_t}"] = data_RI_trott_deltat[:, 2]
    globals()[f"sz_RI_trott_{delta_t}"] = data_RI_trott_deltat[:, 3]

data_RI_notrott = np.loadtxt("RI_solution_notrott.txt")
x_RI_notrott  = data_RI_notrott[:, 0]*g
sx_RI_notrott= data_RI_notrott[:, 1]
sy_RI_notrott = data_RI_notrott[:, 2]
sz_RI_notrott = data_RI_notrott[:, 3]

data_qiskit_kingston= np.loadtxt("qiskit2_kingston_sxsysz_ns1024.txt")
        
x_qiskit_kingston  = data_qiskit_kingston[:, 0]*g
sx_qiskit_kingston= data_qiskit_kingston[:, 1]
sy_qiskit_kingston = data_qiskit_kingston[:, 2]
sz_qiskit_kingston = data_qiskit_kingston[:, 3]

data_qiskit_fez= np.loadtxt("qiskit2_fez_sxsysz_ns1024.txt")
        
x_qiskit_fez  = data_qiskit_fez[:, 0]*g
sx_qiskit_fez= data_qiskit_fez[:, 1]
sy_qiskit_fez = data_qiskit_fez[:, 2]
sz_qiskit_fez = data_qiskit_fez[:, 3]

data_qiskit_marrakesh= np.loadtxt("qiskit2_marrakesh_sxsysz_ns1024.txt")
        
x_qiskit_marrakesh  = data_qiskit_marrakesh[:, 0]*g
sx_qiskit_marrakesh= data_qiskit_marrakesh[:, 1]
sy_qiskit_marrakesh = data_qiskit_marrakesh[:, 2]
sz_qiskit_marrakesh = data_qiskit_marrakesh[:, 3]

data_qiskit_nonoise = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024.txt")
        
x_qiskit_nonoise  = data_qiskit_nonoise[:, 0]*g
sx_qiskit_nonoise= data_qiskit_nonoise[:, 1]
sy_qiskit_nonoise = data_qiskit_nonoise[:, 2]
sz_qiskit_nonoise = data_qiskit_nonoise[:, 3]

data_qiskit_nonoise_swap = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024_swaptrott.txt")
        
x_qiskit_nonoise_swap  = data_qiskit_nonoise_swap[:, 0]*g
sx_qiskit_nonoise_swap= data_qiskit_nonoise_swap[:, 1]
sy_qiskit_nonoise_swap = data_qiskit_nonoise_swap[:, 2]
sz_qiskit_nonoise_swap = data_qiskit_nonoise_swap[:, 3]

data_qiskit_nonoise_wA_05 = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024_wA_0.5.txt")
        
x_qiskit_nonoise_wA_05  = data_qiskit_nonoise_wA_05[:, 0]*g
sx_qiskit_nonoise_wA_05= data_qiskit_nonoise_wA_05[:, 1]
sy_qiskit_nonoise_wA_05 = data_qiskit_nonoise_wA_05[:, 2]
sz_qiskit_nonoise_wA_05 = data_qiskit_nonoise_wA_05[:, 3]

data_qiskit_nonoise_wA_1 = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024_wA_1.txt")
        
x_qiskit_nonoise_wA_1  = data_qiskit_nonoise_wA_1[:, 0]*g
sx_qiskit_nonoise_wA_1= data_qiskit_nonoise_wA_1[:, 1]
sy_qiskit_nonoise_wA_1 = data_qiskit_nonoise_wA_1[:, 2]
sz_qiskit_nonoise_wA_1 = data_qiskit_nonoise_wA_1[:, 3]

data_qiskit_nonoise_wA_10 = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024_wA_10.txt")
        
x_qiskit_nonoise_wA_10  = data_qiskit_nonoise_wA_10[:, 0]*g
sx_qiskit_nonoise_wA_10= data_qiskit_nonoise_wA_10[:, 1]
sy_qiskit_nonoise_wA_10 = data_qiskit_nonoise_wA_10[:, 2]
sz_qiskit_nonoise_wA_10 = data_qiskit_nonoise_wA_10[:, 3]

data_qiskit_nonoise_Nfree_30 = np.loadtxt("qiskit2_nonoise_sxsysz_ns1024_Nfree_30.txt")
        
x_qiskit_nonoise_Nfree_30  = data_qiskit_nonoise_Nfree_30[:, 0]*g
sx_qiskit_nonoise_Nfree_30= data_qiskit_nonoise_Nfree_30[:, 1]
sy_qiskit_nonoise_Nfree_30 = data_qiskit_nonoise_Nfree_30[:, 2]
sz_qiskit_nonoise_Nfree_30 = data_qiskit_nonoise_Nfree_30[:, 3]


data_qiskit_fez_Nfree_30 = np.loadtxt("qiskit2_fez_sxsysz_ns1024_Nfree_30.txt")
        
x_qiskit_fez_Nfree_30  = data_qiskit_fez_Nfree_30[:, 0]*g
sx_qiskit_fez_Nfree_30= data_qiskit_fez_Nfree_30[:, 1]
sy_qiskit_fez_Nfree_30 = data_qiskit_fez_Nfree_30[:, 2]
sz_qiskit_fez_Nfree_30 = data_qiskit_fez_Nfree_30[:, 3]
# %% Noise vs no noise
plt.plot(x_cont, sx_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
plt.plot(x_cont, sy_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
plt.plot(x_cont, sz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')

plt.scatter(
            x_qiskit_nonoise, sx_qiskit_nonoise,
            marker='o', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ Qiskit (no noise)'
        )


plt.scatter(
    x_qiskit_nonoise, sy_qiskit_nonoise,
    marker='o', edgecolors='orange',
    label=rf'$\langle \sigma_y\rangle$ Qiskit (no noise)'
)

plt.scatter(
    x_qiskit_nonoise, sz_qiskit_nonoise,
        marker='o', edgecolors='green',
        label=rf'$\langle \sigma_z\rangle$ Qiskit (no noise)'
)

plt.scatter(
            x_qiskit_marrakesh, sx_qiskit_marrakesh,
            marker='^', facecolors='none', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ Qiskit IBM_marrakesh'
        )


plt.scatter(
    x_qiskit_marrakesh, sy_qiskit_marrakesh,
    marker='^', facecolors='none', edgecolors='orange',
    label=rf'$\langle \sigma_y\rangle$ Qiskit IBM_marrakesh'
)

plt.scatter(
    x_qiskit_marrakesh, sz_qiskit_marrakesh,
        marker='^', facecolors='none', edgecolors='green',
        label=rf'$\langle \sigma_z\rangle$ Qiskit IBM_marrakesh'
)
plt.xlabel('gt')
plt.ylabel('Expected Value')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.show()

# %% Swap terms trott

plt.plot(x_cont, sx_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
plt.plot(x_cont, sy_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
plt.plot(x_cont, sz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')

plt.scatter(
            x_qiskit_nonoise, sx_qiskit_nonoise,
            marker='o', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $U = e^{iH\Delta t}e^{iV\Delta t}$'
        )


plt.scatter(
    x_qiskit_nonoise, sy_qiskit_nonoise,
    marker='o', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $U = e^{iH\Delta t}e^{iV\Delta t}$'
)

plt.scatter(
    x_qiskit_nonoise, sz_qiskit_nonoise,
        marker='o', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $U = e^{iH\Delta t}e^{iV\Delta t}$'
)


plt.scatter(
            x_qiskit_nonoise_swap, sx_qiskit_nonoise_swap,
            marker='^', facecolors='none', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $U = e^{iV\Delta t}e^{iH\Delta t}$'
        )


plt.scatter(
    x_qiskit_nonoise_swap, sy_qiskit_nonoise_swap,
    marker='^', facecolors='none', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $U = e^{iV\Delta t}e^{iH\Delta t}$'
)

plt.scatter(
    x_qiskit_nonoise_swap, sz_qiskit_nonoise_swap,
        marker='^', facecolors='none', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $U = e^{iV\Delta t}e^{iH\Delta t}$'
)

plt.title('Qiskit No noise')
plt.xlabel('gt')
plt.ylabel('Expected Value')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.show()
# %%Add wA
plt.plot(x_cont, sx_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
plt.plot(x_cont, sy_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
plt.plot(x_cont, sz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')

plt.scatter(
            x_qiskit_nonoise, sx_qiskit_nonoise,
            marker='o', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $\omega_a = 0$'
        )


plt.scatter(
    x_qiskit_nonoise, sy_qiskit_nonoise,
    marker='o', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $\omega_a = 0$'
)

plt.scatter(
    x_qiskit_nonoise, sz_qiskit_nonoise,
        marker='o', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $\omega_a = 0$'
)


plt.scatter(
            x_qiskit_nonoise_wA_05, sx_qiskit_nonoise_wA_05,
            marker='^', facecolors='none', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $\omega_a = 0.5$'
        )


plt.scatter(
    x_qiskit_nonoise_wA_05, sy_qiskit_nonoise_wA_05,
    marker='^', facecolors='none', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $\omega_a = 0.5$'
)

plt.scatter(
    x_qiskit_nonoise_wA_05, sz_qiskit_nonoise_wA_05,
        marker='^', facecolors='none', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $\omega_a = 0.5$'
)

plt.scatter(
            x_qiskit_nonoise_wA_1, sx_qiskit_nonoise_wA_1,
            marker='o', facecolors='none', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $\omega_a = 1$'
        )


plt.scatter(
    x_qiskit_nonoise_wA_1, sy_qiskit_nonoise_wA_1,
    marker='o', facecolors='none', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $\omega_a = 1$'
)

plt.scatter(
    x_qiskit_nonoise_wA_1, sz_qiskit_nonoise_wA_1,
        marker='o', facecolors='none', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $\omega_a = 1$'
)

plt.scatter(
            x_qiskit_nonoise_wA_10, sx_qiskit_nonoise_wA_10,
            marker='s', facecolors='none', edgecolors='blue',
            label=r'$\langle \sigma_x\rangle$ Qiskit $\omega_a = 10$'
        )


plt.scatter(
    x_qiskit_nonoise_wA_10, sy_qiskit_nonoise_wA_10,
    marker='s', facecolors='none', edgecolors='orange',
    label=r'$\langle \sigma_y\rangle$ Qiskit $\omega_a = 10$'
)

plt.scatter(
    x_qiskit_nonoise_wA_10, sz_qiskit_nonoise_wA_10,
        marker='s', facecolors='none', edgecolors='green',
        label=r'$\langle \sigma_z\rangle$ Qiskit $\omega_a = 10$'
)

plt.title(r'Qiskit change $\omega_A$')
plt.xlabel('gt')
plt.ylabel('Expected Value')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.show()
# %% Free evolution Nfree
plt.plot(x_cont, sx_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
plt.plot(x_cont, sy_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
plt.plot(x_cont, sz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')

plt.scatter(
            x_qiskit_fez_Nfree_30, sx_qiskit_fez_Nfree_30,
            marker='o', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ Qiskit IBM_fez + free ev'
        )


plt.scatter(
    x_qiskit_fez_Nfree_30, sy_qiskit_fez_Nfree_30,
    marker='o', edgecolors='orange',
    label=rf'$\langle \sigma_y\rangle$ Qiskit IBM_fez + free ev'
)

plt.scatter(
    x_qiskit_fez_Nfree_30, sz_qiskit_fez_Nfree_30,
        marker='o', edgecolors='green',
        label=rf'$\langle \sigma_z\rangle$ Qiskit IBM_fez + free ev'
)

plt.title('Adding free evolution of the system IBM_fez (Nfree = 30)')
plt.xlabel('gt')
plt.ylabel('Expected Value')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.show()

# %% Trot vs no Trott
plt.plot(x_cont, sx_cont, label=rf'$\langle \sigma_x\rangle$ Cont')
plt.plot(x_cont, sy_cont, label=rf'$\langle \sigma_y\rangle$ Cont')
plt.plot(x_cont, sz_cont, label=rf'$\langle \sigma_z\rangle$ Cont')

plt.scatter(
            x_RI_trott, sx_RI_trott,
            marker='o', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ RI  trott'
        )


plt.scatter(
    x_RI_trott, sy_RI_trott,
    marker='o', edgecolors='orange',
    label=rf'$\langle \sigma_y\rangle$ RI trott'
)

plt.scatter(
    x_RI_trott, sz_RI_trott,
        marker='o', edgecolors='green',
        label=rf'$\langle \sigma_z\rangle$ RI  trott'
)

plt.scatter(
            x_RI_notrott, sx_RI_notrott,
            marker='^', facecolors='none', edgecolors='blue',
            label=rf'$\langle \sigma_x\rangle$ RI no trott'
        )


plt.scatter(
    x_RI_notrott, sy_RI_notrott,
    marker='^', facecolors='none', edgecolors='orange',
    label=rf'$\langle \sigma_y\rangle$ RI no trott'
)

plt.scatter(
    x_RI_notrott, sz_RI_notrott,
        marker='^', facecolors='none', edgecolors='green',
        label=rf'$\langle \sigma_z\rangle$ RI no trott'
)
plt.xlabel('gt')
plt.ylabel('Expected Value')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.show()
# %% Errors trott vs no trott
from scipy.interpolate import interp1d

interp_cont_z = interp1d(
        x_cont, sz_cont,
        bounds_error=True
    )
interp_cont_x = interp1d(
        x_cont, sx_cont,
        bounds_error=True
    )
interp_cont_y = interp1d(
        x_cont, sy_cont,
        bounds_error=True
    )
error_trott_sx = np.sqrt(np.mean((sx_RI_trott - interp_cont_x(x_RI_trott))**2))

error_trott_sy = np.sqrt(np.mean((sy_RI_trott - interp_cont_y(x_RI_trott))**2))
error_trott_sz = np.sqrt(np.mean((sz_RI_trott - interp_cont_z(x_RI_trott))**2))

error_notrott_sx = np.sqrt(np.mean((sx_RI_notrott - interp_cont_x(x_RI_notrott))**2))
error_notrott_sy = np.sqrt(np.mean((sy_RI_notrott - interp_cont_y(x_RI_notrott))**2,))
error_notrott_sz = np.sqrt(np.mean((sz_RI_notrott - interp_cont_z(x_RI_notrott))**2))

print('RMSE no trott sx:',error_notrott_sx)
print('RMSE no trott sy:',error_notrott_sy)
print('RMSE no trott sz:',error_notrott_sz)
print('Total error no trott: ',np.sqrt(error_notrott_sx**2 + error_notrott_sy**2 + error_notrott_sz**2))

print('RMSE trott sx:',error_trott_sx)
print('RMSE trott sy:',error_trott_sy)
print('RMSE trott sz:',error_trott_sz)
print('Total error trott: ',np.sqrt(error_trott_sx**2 + error_trott_sy**2 + error_trott_sz**2))


# %% Change Delta t

for delta_t in list_deltat:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Solución continua: líneas
    x_cont = globals()[f"x_cont_{delta_t}"]
    ax.plot(x_cont, globals()[f"sx_cont_{delta_t}"], '-', label=r"$s_x$ cont")
    ax.plot(x_cont, globals()[f"sy_cont_{delta_t}"], '-', label=r"$s_y$ cont")
    ax.plot(x_cont, globals()[f"sz_cont_{delta_t}"], '-', label=r"$s_z$ cont")

    # Solución RI Trott: scatter
    x_RI = globals()[f"x_RI_trott_{delta_t}"]
    ax.scatter(x_RI, globals()[f"sx_RI_trott_{delta_t}"], s=15, label=r"$s_x$ RI")
    ax.scatter(x_RI, globals()[f"sy_RI_trott_{delta_t}"], s=15, label=r"$s_y$ RI")
    ax.scatter(x_RI, globals()[f"sz_RI_trott_{delta_t}"], s=15, label=r"$s_z$ RI")

    ax.set_xlabel(r"$gt$")
    ax.set_ylabel(r"Expected value")
    ax.set_title(fr"$\Delta t = {delta_t}$")
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.show()
# %%
print(x_RI_trott)
print(x_RI_notrott)
# %%
