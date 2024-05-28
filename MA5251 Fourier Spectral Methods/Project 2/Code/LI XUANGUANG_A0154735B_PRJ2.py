#%%
'''

MA5251 Project 2
Creator: Li Xuanguang
Student ID: A0154735B

Burgers Equation Solver

'''
#%%

####################### IMPORT MODULES ########################
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

#%%

# Class function to run Burger's Equation program
class burger:

    # Initialise the class
    # Inputs: 
    def __init__(self, N, T, dt, dtp):
        self.N = N
        self.T = T
        self.dt = dt
        self.dtp = dtp

    '''
    Initialisation of values functions
    '''

    def Diff1D(self, u):

        u = np.array(u, dtype=np.float128)
        Du = np.zeros(np.shape(u))      
        N = u.shape[0]
        Du[N-2] = 2*(N-1)*u[N-1]
        for i in reversed(range(2,N-1)):
            Du[i-1] = Du[i+1] + 2 * i * u[i]
        
        return Du
    
    def DChebT1D(self, u):
    
        u = np.array(u, dtype=np.float128)
        u[0] = u[0] / np.sqrt(2)
        u[-1] = u[-1] / np.sqrt(2)
        U = scipy.fftpack.dct(u, norm='ortho', type = 1) * np.sqrt(2 / (u.shape[0]-1))
        U[0] = U[0] / np.sqrt(2)
        U[-1] = U[-1] / np.sqrt(2)

        return U
    
    def IDChebT1D(self, u):
    
        u = np.array(u, dtype=np.float128)
        u[0] = u[0] * np.sqrt(2)
        u[-1] = u[-1] * np.sqrt(2)
        U = scipy.fftpack.dct(u, norm='ortho', type = 1) / np.sqrt(2 / (u.shape[0]-1))
        U[0] = U[0] * np.sqrt(2)
        U[-1] = U[-1] * np.sqrt(2)

        return U
    
    def rhs(self, u):

        u = np.array(u, dtype=np.float128)
        du = self.IDChebT1D(self.Diff1D(u))
        u_square = np.power(self.IDChebT1D(u),2)
        f = self.Diff1D(self.DChebT1D(np.multiply(du, 0.02) - np.multiply(u_square, 0.5)))

        return f
    
    def set_bc(self, u, a, b, t):

        u = np.array(u, dtype=np.float128)
        uv = self.IDChebT1D(u)
        uv[0] = b(t)
        uv[-1] = a(t)
        u = self.DChebT1D(uv)

        return u

   
    def create_plots(self, curve, time):
        
        x = np.linspace(-1, 1, 100)
        poly_coefs = curve[::-1]
        y = np.polyval(poly_coefs, x)
        fig = plt.figure()
        ax = plt.plot(x, y)

        # Set directory
        cwd = os.getcwd()
        # Save file directory
        plotsLoc = 'plots'
        plotsLoc = os.path.join(cwd, plotsLoc)
        # Save file location
        fig_name = "t" + str(round(float(time), 1)) + '_plot' + '.png'
        print('Creating ' + fig_name + ' plot for time ' + str(round(float(time), 1)))
        figLoc = os.path.join(plotsLoc, fig_name)
        # Save the file
        plt.savefig(figLoc)
        plt.close()
    
    def burger1D(self, u0, a, b):

        # Collocation Points
        col_points = np.cos(np.arange(self.N+1)*np.pi/(self.N))
        
        # Initial Value
        u = self.DChebT1D(u0(col_points))

        # Iteration
        t = 0
        tp = self.dtp - (self.dt/2)

        # To print the results
        if t == 0:

            cheb = np.polynomial.Chebyshev(u)
            poly = cheb.convert(kind=np.polynomial.Polynomial, domain=[-1, 1])
            self.create_plots(poly.convert().coef, t)
        
        while t < self.T:

            if t + self.dt > self.T:
                dt = self.T-t
        
            # SSP-RK3
            u1 = u + self.dt*self.rhs(u)
            u1 = self.set_bc(u1, a, b, t + self.dt)
            u1 = 0.25 * (u1 + self.dt * self.rhs(u1)) + 0.75*u
            u1 = self.set_bc(u1, a, b, t + (self.dt/2))
            u = (2/3)*(u1 + self.dt*self.rhs(u1)) + (1/3)*u
            u = self.set_bc(u, a, b, t + self.dt)

            t += self.dt

            # To print the results
            if t >= tp:

                cheb = np.polynomial.Chebyshev(u)
                poly = cheb.convert(kind=np.polynomial.Polynomial, domain=[-1, 1])
                self.create_plots(poly.convert().coef, t)

                tp = tp + self.dtp

    def run(self):
        u0 = lambda x: -np.sin(5*np.pi*x/2)
        a = lambda t: 1
        b = lambda t: -1
        self.burger1D(u0, a, b)


#%%

'''
Running the script with parameters
'''

if __name__ == '__main__':
    burger_calc = burger(32, 5, 0.001, 0.3)
    results = burger_calc.run()


# %%
