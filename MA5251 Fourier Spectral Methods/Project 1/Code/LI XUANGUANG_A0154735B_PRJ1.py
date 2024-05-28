#%%
'''

MA5251 Project 1
Creator: Li Xuanguang
Student ID: A0154735B

Schrodinger Equation Solver

'''
#%%

####################### IMPORT MODULES ########################
import numpy as np
import scipy.fft
import matplotlib.pylab as plt
from matplotlib.ticker import LinearLocator
import timeit
import os
from matplotlib import cm


#%%

# Class function to run Schrodinger's Equation program
class schrod:

    # Initialise the class
    # Inputs: timesteps (int), time range (absolute value, int) - 0 to upper second?, resolution (int), coordinate range (absolute value, int)
    def __init__(self, timestep, t_upper, coord, mult = 1):
        self.gamma = [2,4] # gamma_2, gamma_3
        self.varepsilon = 1/4
        self.dt = 1/timestep # size of each time step
        self.timesteps = int(timestep * t_upper) # number of time steps to compute
        self.coord = coord # range of coordinates centered on zero to initialise
        self.mult = mult # multiple of pi range to use
        self.t_upper = t_upper # the upper limit of seconds to compute
        self.cent_1, self.cent_1sq, self.cent_2 = self.centre_array() # Creates array centered around 0
        self.xx, self.yy, self.zz = self.meshgrid_calc_3d() # Create the coordinate t_0 meshgrid
        self.x1, self.y1, self.z1 = self.meshgrid_calc_2d() # Create the values t_0 meshgrid
        self.psi_x = self.init_psi_x()
        self.psi_t0 = self.init_psi_t0()
        self.v_x = self.init_V_x()
        self.mu = self.mu_val()


    '''
    Initialisation of values functions
    '''

    # Function to create default 3d meshgrid for coordinates
    # Input: dimensions (int)
    # Output; meshgrid
    def meshgrid_calc_3d(self):

        print('Creating a coordinate meshgrid')

        # Create linearspace - V(X_j) ranges from (0, ..., N-1)^n
        # Ensure linearspace ranges from 0 to 2 pi
        val_range = (2*np.pi*(self.coord-1))/self.coord
        a = np.linspace(0, val_range, self.coord)

        xx, yy, zz = np.meshgrid(a, a, a)
        return xx, yy, zz
    
    # Function to create 2d grid with z for ref for values
    # Input: np array of k^2 values
    # Output; meshgrid
    def meshgrid_calc_2d(self):

        print('Creating a value meshgrid')

        # Create the meshgrid
        x1, y1 = np.meshgrid(self.cent_1sq, self.cent_1sq)
        z1 = np.ones((self.coord, self.coord), dtype = 'complex_')

        return x1, y1, z1
    
    # Function create array of values for N, centered around 0
    # Input: size N
    # Output; array centered around 0
    def centre_array(self):
    
        # simplifies weighting for FFT calculations
        n_half = int(self.coord/2)
        cent_one = np.concatenate((list(range(0,n_half+1)),list(range(-n_half+1, 0))))
        cent_one_sq = cent_one**2
        cent_two = np.concatenate((list(range(0,n_half)),list(range(-n_half, 0))))
        
        return cent_one, cent_one_sq, cent_two

        
    # Function to initialise the potential function
    # Note, function is static, create once only
    # Output: V_x meshgrid
    def init_V_x(self):
        
        print('Initialising V_x on meshgrid')

        v_x_val = 1/2 * (self.mult**2) * (((self.xx-np.pi)**2) + (self.gamma[0]**2)*((self.yy-np.pi)**2) + (self.gamma[1]**2)*((self.zz-np.pi)**2))
        
        return v_x_val

    # Function to initialise the psi(x,0) function
    # Output: psi_x_0 meshgrid
    def init_psi_x(self):

        print('Initialising Psi_x_0 on meshgrid')

        exp_interior = -((self.mult**2)*(self.xx-np.pi)**2 + self.gamma[0]*((self.yy-np.pi)**2) + self.gamma[1]*((self.zz-np.pi)**2))/(2*self.varepsilon)

        exp_exterior = ((self.gamma[0]*self.gamma[1])**(1/4))/((np.pi * self.varepsilon)**(3/4))

        psi_value = exp_exterior * np.exp(exp_interior)

        return psi_value
    
    # Function to fourier transform the psi(x,0) function
    # Output: psi_t=0
    def init_psi_t0(self):

        psi_t0 = scipy.fft.fftn(self.psi_x)

        return psi_t0
    
    # Function to simplify calcuation of mu in exp RK method
    # Output: mu
    def mu_val(self):

        mu = np.exp(1j*3/2)

        return mu
    
    '''
    Fourier Transform Functions
    '''
    
    # Function to compute the nonlinear term (|psi|^2) for one value of k
    # Inputs: u_k{time = t}, k = 0, ..., 8 (time)
    # Output: u_k{time = t}^2
    def nonlinear_compute(self, u_k_t0, k):

        n_half = int(self.coord/2)

        # Create NxNxN array to store results
        u_k_n = np.zeros((self.coord, self.coord, self.coord), dtype = 'complex_')

        # As end points of psi are half, we need to set up the values accordingly
        psi_sq = np.abs(u_k_t0**2)

        psi_sq[n_half,:,k] = (1/2)*psi_sq[n_half,:,k]
        psi_sq[:,n_half,k] = (1/2)*psi_sq[:,n_half,k]

        if k == n_half:
            psi_sq[:,:,k] = (1/2)*psi_sq[:,:,k]

        # Store in results df
        u_k_n[:,:,k] = u_k_n[:,:,k] + ((2*np.pi/(self.coord**2))**3)*(self.mult**3)*(1/10)*np.sum(psi_sq)*u_k_t0[:,:,k]
        
        return u_k_n
    
    # Function to compute the inear term (V(x) * psi) for one value of k
    # Inputs: u_k_{time=t}, u_k_n{time = t}, k = 0, ..., 8, time = t
    # Output: u_k_n{time = t}^2
    def linear_compute(self, u_k_t0, u_k_n, k):

        n_half = int(self.coord/2)

        u_k_n_t0 = scipy.fft.ifftn(u_k_t0)

        # Create the correct weightage of terms
        #self.cent_1 = mm,  self.cent_2 = mm1
        #self.xx, self.yy, self.zz
        for n in range(0,n_half):

            for p in range(0, self.coord):

                weight_df_LHS = (1/2)*np.exp((-1j)*self.cent_1[n]*self.xx)*np.exp((-1j)*self.cent_1[p]*self.yy)*np.exp((-1j)*self.cent_1[k]*self.zz)
                weight_df_RHS = (1/2)*np.exp((-1j)*self.cent_2[n]*self.xx)*np.exp((-1j)*self.cent_2[p]*self.yy)*np.exp((-1j)*self.cent_2[k]*self.zz)

                weight_df = weight_df_LHS + weight_df_RHS

                # Edge cases

                if k == n_half+1:
                    weight_df[:, :,k] == 2*weight_df[:, :,k]

                if n == n_half+1:
                    weight_df[: ,n, :] == 2*weight_df[: ,n, :]

                if p == n_half+1:
                    weight_df[p,:,:] == 2*weight_df[p,:,:]
                
                # Compute the result
                term = (1/(self.coord**3))*(self.v_x)*weight_df*u_k_n_t0

                # Save value in u_k_n
                u_k_n[p,n,k] = u_k_n[p,n,k] + np.sum(term)

        return u_k_n
    

    '''
    Exponential Runge-Kutta Method
    '''

    
    # Function to compute one step of Exponential Runge Kutta Order 1 Method
    # Inputs: u_k_t0{time = t} (time), computed u_k_n which is F(u) (spatial)
    # Output: u_t{time = t+1} (time)
    def exp_rk_1storder(self, u_k_t0, u_k_n, time):

        print("Calculating First Order Runge-Kutta for timestep " + str(time))

        u_k_tp1 = u_k_n.copy() # to ensure no changes to u_k_n as k varies

        for k in range(0, self.coord):
            u_k_t1_rhs = u_k_t0[:, :,k] - 1j*self.dt*u_k_n[:, :,k]
            u_k_t1_lhs = np.exp(-self.mu*self.dt*(self.x1 + self.y1 + self.z1))
            u_k_t1 = u_k_t1_lhs*u_k_t1_rhs

            # Store values 
            u_k_tp1[:,:,k] = u_k_t1

        return u_k_tp1

    # Function to compute one step of Exponential Runge Kutta Order 3 Method
    # psi_x_t is the calculated spatial term
    # u_k_n is taken from x2t_1stterm(psi_x_t)
    # v_x is the time-independent potential
    def exp_rk_3rdorde(self, psi_x_t, u_k_n, v_x, time):

        print("Calculating Third Order Runge-Kutta for timestep " + str(time))

        # u_k_(1)
        u_k_1_rhs = u_k_n + (2/3)*self.dt*self.f_k_un(psi_x_t)
        u_k_1 = self.exp_func(2/3) * u_k_1_rhs

        # u_k_(2)
        u_k_2_rhs = (1/3)*(u_k_1 + (4/3)*self.dt*self.f_k_un(u_k_1))
        u_k_2 = (2/3)*self.exp_func(2/3)*u_k_n + u_k_2_rhs

        # u_k_(n+1)
        u_k_n1_1st = (59/128)*self.exp_func(1)*u_k_n
        u_k_n1_2nd = (15/128)*self.exp_func(1)*(u_k_n + (4/3)*self.dt*self.f_k_un(psi_x_t))
        u_k_n1_3rd = (27/64)*self.exp_func(1/3)*(u_k_2 + (4/3)*self.dt*self.f_k_un(u_k_2))
        u_k_n1 = u_k_n1_1st + u_k_n1_2nd + u_k_n1_3rd

        u_k_n1 = scipy.fft.ifftn(u_k_n1)

        return u_k_n1 # This is psi_x_{t+1}
    
    '''
    Graphing Tools
    '''

    # Function to flatten by 1 dimension (remove z dimension)
    # Input: f as numpy array
    # Output: f with x, y only
    def flatten_z(self, f):
        
        # Shift z to the front for integration
        f = np.array(f).transpose(0, 1, 2)
        numeric_int_arr = np.empty(self.coord, dtype = 'complex_')
        numeric_int_arr.fill(1)
        result = np.array(f).dot(numeric_int_arr)

        return result
    
    
    # Function to plot the surface
    # Input: f as numpy array, func = 'angle' or 'abs' or 'imag' or 'real',time (timestamp) (float)
    # Output: surface plot
    def create_surface_plots(self, data, func, time):

        val_range = (2*np.pi*(self.coord-1))/self.coord
        a = np.linspace(0, val_range, self.coord)

        xx, yy = np.meshgrid(a, a)

        f = self.flatten_z(data)
        g = np.real(f)
        if func == 'angle':
            g = np.angle(f)
        elif func == 'abs':
            g = np.abs(f)
        elif func == 'imag':
            g = np.imag(f)
        else:
            func = 'real'
            g = np.real(f)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(xx, yy, g, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        
        # Customize the z axis.
        ax.set_zlim(-3, 3)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Set directory
        cwd = os.getcwd()
        # Save file directory
        plotsLoc = 'surfaceplots'
        plotsLoc = os.path.join(cwd, plotsLoc)
        saveFile = os.path.join(plotsLoc, func)
        # Save file location
        fig_name = "t" + str(round(float(time), 10)) + '_' + func + '_surfaceplot' + '.png'
        print('Creating ' + fig_name + ' plot for ' + func + ' for time ' + str(time))
        figLoc = os.path.join(saveFile, fig_name)
        # Save the file
        plt.savefig(figLoc)
        plt.close()

    
    # Function to plot the 3d scatter
    # Input: f as numpy array, func = 'angle' or 'abs' or 'imag' or 'real',time (timestamp) (float)
    # Output: 3d scatter
    def create_3d_plots(self, data, func, time):

        g = np.real(data)
        if func == 'angle':
            g = np.angle(data)
        elif func == 'abs':
            g = np.abs(data)
        elif func == 'imag':
            g = np.imag(data)
        else:
            func = 'real'
            g = np.real(data)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(schrod_calc.xx, schrod_calc.yy, schrod_calc.zz, c=g, cmap='Purples', alpha=0.7, marker='.')

        # Set directory
        cwd = os.getcwd()
        # Save file directory
        plotsLoc = '3dplots'
        plotsLoc = os.path.join(cwd, plotsLoc)
        saveFile = os.path.join(plotsLoc, func)
        # Save file location
        fig_name = "t" + str(round(float(time), 10)) + '_' + func + '_3dplot' + '.png'
        print('Creating ' + fig_name + ' plot for ' + func + ' for time ' + str(time))
        figLoc = os.path.join(saveFile, fig_name)
        # Save the file
        plt.savefig(figLoc)
        plt.close()

    # Function to plot all plots
    # Input: f as numpy array, func = 'angle' or 'abs' or 'imag' or 'real',time (timestamp) (float)
    # Output: 3d scatter
    def plotall(self, data, time):

        # Create the surface plots
        self.create_surface_plots(data, 'real', time)
        self.create_surface_plots(data, 'angle', time)
        self.create_surface_plots(data, 'abs', time)
        self.create_surface_plots(data, 'imag', time)

        # Create the 3d plots
        self.create_3d_plots(data, 'real', time)
        self.create_3d_plots(data, 'angle', time)
        self.create_3d_plots(data, 'abs', time)
        self.create_3d_plots(data, 'imag', time)


    '''
    Main section to run the script
    '''

    def run(self):

        # To track run time
        start = timeit.timeit()

        # Storage for results
        psi_x_t_results = [self.psi_x]

        # Initialise the data
        timenow = 0 # time initially at 0

        # Plot the first datapoint
        self.plotall(self.psi_x, timenow)
        
        # Both linear and nonlinear parts are required to compute a complete f_k(u_k_n(t=0))
        for k in range(0, self.coord):
            u_k_n_te0 = self.nonlinear_compute(self.psi_t0, k)
            u_k_n_te0 = self.linear_compute(self.psi_t0, u_k_n_te0, k)

        # we feed the f_k(u_k_n(t=0)) into the exp rk to compute the next step
        u_k_t1 = self.exp_rk_1storder(self.psi_t0, u_k_n_te0, timenow) # Result is equivalent to u^n{t=1}

        # Then we ifft and store the data
        u_k_n_te1 = scipy.fft.ifftn(u_k_t1) # Result is u_k_n(t=1)
        psi_x_t_results += [u_k_n_te1]

        # Store the results
        timenow += self.dt
        timenow = round(timenow, 2)

        # Plot the next datapoint
        self.plotall(u_k_n_te1, timenow)

        # Set u_k{t=i} to u_k{t=1} to begin the run
        psi_ti = u_k_t1

        # Loop over time range
        for t in range(1, self.timesteps):

            timenow += self.dt
            timenow = round(timenow, 2)

            if t % 10 == 0:
                print('Computing for timestep = ' + str(t+1) + ', currently at time = ' + str(timenow))

            for k in range(0, self.coord):
                # Both linear and nonlinear parts are required to compute a complete f_k(u_k_n(t=i))
                u_k_n_tei = self.nonlinear_compute(psi_ti, k)
                u_k_n_tei = self.linear_compute(psi_ti, u_k_n_tei, k)

            # we feed the f_k(u_k_n(t=i)) into the exp rk to compute the next step
            u_k_tip1 = self.exp_rk_1storder(psi_ti, u_k_n_tei, timenow) # Result is equivalent to u^n{t=i+1}

            # Then we ifft and store the data
            u_k_n_teip1 = scipy.fft.ifftn(u_k_tip1)
            psi_x_t_results += [u_k_n_teip1]

            # Plot the datapoint
            self.plotall(u_k_n_teip1, timenow)

            # Update the datapoint
            psi_ti = u_k_tip1



        # Track time
        end = timeit.timeit()
        timeelapsed = end - start
        print('Time Elapsed: ' + str(timeelapsed))

        return psi_x_t_results



#%%

'''
Running the script with parameters
'''

if __name__ == '__main__':
    schrod_calc = schrod(100, 40, 20)
    results = schrod_calc.run()

# %%
