#%%
'''

MA5232 Part 2 - Project
Creator: Li Xuanguang
Student ID: A0154735B

BGK Solver built on discrete maxwellian

'''
#%%

####################### IMPORT MODULES ########################
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import math
from math import pi
import seaborn as sns
import os
import timeit

#%%

# Class function to run BGK program
class BGK:

    # Initialise the class
    # Inputs: step (int)
    def __init__(self, step, lower, upper):
        self.mass = np.array([1, 2]) # mass of single gas molecule of [1st, 2nd] species
        self.k_b = 1 # boltzmann constant
        self.v_ij = np.array([[1, 3/2], [3/2, 2]]) # constant given
        self.number_density = np.array([2, 1]) # number density of gas molecules of [1st, 2nd] species 
        self.density = np.multiply(self.mass, self.number_density) # density of gas molecules of [1st, 2nd] species 
        self.u_i_0 = np.array([[-1,0,0], [1,0,0]]) # velocity of [1st, 2nd] species at time 0
        self.t_i_0 = np.array([1, 2]) # temperature of [1st, 2nd] species at time 0
        self.u_ij_0 = np.array(self.u_ij_0_calc())
        self.t_ij_0 = np.array(self.t_ij_0_calc())
        self.step = step # Step
        self.lower = lower # lower limit
        self.upper = upper # upper limit
        self.xx, self.yy, self.zz = self.meshgrid_calc(lower, upper, 3) # Create the meshgrid
        self.weight = self.weights()

    '''
    Initialisation of values functions
    '''

    # Function to create meshgrid for calculations
    # Input: lower and uppper bounds (both int), dimensions (int)
    # Output; meshgrid
    def meshgrid_calc(self, lower, upper, dim):

        print('Creating a meshgrid')

        # Create linearspace
        a = np.linspace(lower, upper, self.step)
        b = np.linspace(lower, upper, self.step)
        c = np.linspace(lower, upper, self.step)

        if dim == 1:
            
            return a

        elif dim == 2:
            # Create the meshgrid
            xx, yy = np.meshgrid(a, b)
            return xx, yy
        
        elif dim == 3:
            # Create the meshgrid
            xx, yy, zz = np.meshgrid(a, b, c)
            return xx, yy, zz

    # Function to initialise the distribution function values for each species
    # Input: velocity (3D vector - array)
    # Output: [f_1_0, f_2_0]
    def f_i_0(self):
        
        print('Creating f_i_0 on meshgrid')

        # To save the output
        output_matrix = []

        # For the 2 diff species
        for species in range(1,3):

            # retrieve the values for the species
            n_i = self.number_density[species - 1] # density
            m_i = self.mass[species - 1] # mass
            t_i_zero = self.t_i_0[species - 1] # temperature
            u_ij_zero = self.u_i_0[species - 1] # velocity 
            
            # first half of equation
            first_half = n_i * ((m_i / (2 * pi * self.k_b * t_i_zero)) ** (3/2))
            # second half of equation
            v_minus_u = np.sqrt((u_ij_zero[0]-self.xx)**2 + (u_ij_zero[1]-self.yy)**2 + (u_ij_zero[2]-self.zz)**2)
            second_half = (m_i * (v_minus_u ** 2)) / (2 * self.k_b * t_i_zero)

            # entire equation
            solution = first_half * (np.exp(-second_half))

            # save the output
            output_matrix += [solution]

        return output_matrix
 
    # Function to initialise u_ij_0
    # Input: none
    # Output: computed array at position specified, with time = 0
    # [[u_11, u_12], [u_21, u_22]]
    def u_ij_0_calc(self):

        final_result = []

        # Iterate over i and j combinations
        for species1 in range(1,3):

            l1_result = []

            for species2 in range(1,3):

                # retrieve the values
                p_i_val = self.density[species1 - 1]
                v_ij_val = self.v_ij[species1 - 1][species2 - 1]
                u_i_val = self.u_i_0[species1 - 1]

                p_j_val = self.density[species2 - 1]
                v_ji_val = self.v_ij[species2 - 1][species1 - 1]
                u_j_val = self.u_i_0[species2 - 1]

                # calculate numerator
                num_left = p_i_val * v_ij_val * u_i_val
                num_right = p_j_val * v_ji_val * u_j_val
                # calculate denominator
                denom_left = p_i_val * v_ij_val
                denom_right = p_j_val * v_ji_val
                # calculate result
                result = (num_left + num_right) / (denom_left + denom_right)

                l1_result += [result]
        
            final_result += [l1_result]

        return final_result

    # Function to initialise T_ij_0
    # Input: none
    # Output: computed array at position specified, with time = 0
    # [[T_11, T_12], [T_21, T_22]]
    def t_ij_0_calc(self):

        final_result = []

        # Iterate over i and j combinations
        for species1 in range(1,3):

            l1_result = []

            for species2 in range(1,3):

                # retrieve the values for the species
                n_i_val = self.number_density[species1 - 1]
                v_ij_val = self.v_ij[species1 - 1][species2 - 1]
                t_i_val = self.t_i_0[species1 - 1]

                n_j_val = self.number_density[species2 - 1]
                v_ji_val = self.v_ij[species2 - 1][species1 - 1]
                t_j_val = self.t_i_0[species2 - 1]

                ### Left result
                # calculate numerator
                num_left = n_i_val * v_ij_val * t_i_val
                num_right = n_j_val * v_ji_val * t_j_val
                # calculate denominator
                denom_left = n_i_val * v_ij_val
                denom_right = n_j_val * v_ji_val
                # calculate result
                left_result = (num_left + num_right) / (denom_left + denom_right)


                # retrieve the values for the species
                p_i_val = self.density[species1 - 1]
                u_i_val = self.u_i_0[species1 - 1]
                u_ij_val = self.u_ij_0[species1 - 1][species2 - 1]

                p_j_val = self.density[species2 - 1]
                u_j_val = self.u_i_0[species2 - 1]
                u_ji_val = self.u_ij_0[species2 - 1][species1 - 1]

                ### Right result
                # calculate numerator
                num_left_2 = p_i_val * v_ij_val * (((np.linalg.norm(u_i_val)) ** 2) - ((np.linalg.norm(u_ij_val)) ** 2))
                num_right_2 = p_j_val * v_ji_val * (((np.linalg.norm(u_j_val)) ** 2) - ((np.linalg.norm(u_ji_val)) ** 2))
                denom = 3 * ((n_i_val * v_ij_val) + (n_j_val * v_ji_val))
                # calculate result
                right_result = (num_left_2 + num_right_2) / denom
                
                ## Overall result
                result = left_result + right_result
                # save the value
                l1_result += [result]
            
            final_result += [l1_result]

        return final_result

    # Function to initialise f_eq_ij_v_0
    # Input: velocity (3D vector - array)
    # Output: computed array at position specified, with time = 0
    def f_eq_ij_v_0(self):

        final_result = []

        # Iterate over i and j combinations
        for species1 in range(1,3):

            l1_result = []

            for species2 in range(1,3):

                # retrieve the values
                n_i_value = self.number_density[species1 - 1]
                m_i_value = self.mass[species1 - 1]
                t_ij_0_value = self.t_ij_0[species1 - 1][species2 - 1]
                u_ij_0_value = self.u_ij_0[species1 - 1][species2 - 1]

                # value calculations
                base = n_i_value * ((m_i_value / (2 * pi * self.k_b * t_ij_0_value)) ** (3/2))
                v_minus_u = np.sqrt((u_ij_0_value[0]-self.xx)**2 + (u_ij_0_value[1]-self.yy)**2 + (u_ij_0_value[2]-self.zz)**2)
                expo = (m_i_value * v_minus_u) / (2 * self.k_b * t_ij_0_value)
                result = base * (np.exp(-expo))

                # save the value
                l1_result += [result]

            final_result += [l1_result]

        return final_result

    '''
    Numerical Method Calculations
    '''

    #### Define the fuctions for Forward Euler
    # PDE of f_1
    def df_1(self, f_i, f_eq):
        result = (self.v_ij[0][0] * (f_eq[0][0] - f_i[0])) + (self.v_ij[0][1] * (f_eq[0][1] - f_i[0]))
        return result

    # PDE of f_2
    def df_2(self, f_i, f_eq):
        result = (self.v_ij[1][0] * (f_eq[1][0] - f_i[1])) + (self.v_ij[1][1] * (f_eq[1][1] - f_i[1]))
        return result

    # Function for Forward Euler for a single velocity to calculate next time step t + 1
    # Input: time array [start, end], steps per euler (int)
    # Return: f_i_t+1 value
    def euler(self, f_i_t_value, f_eq_ij_t_value, time, stepsize):

        ti = time[0] # Initial time
        tf = time[1] # Final time
        h = stepsize # Step size
        n = int((tf-ti)/h) # Steps to take
        
        t = ti # Initial value to be specified       

        # Iterate over number of steps to calculate the function value with Euler
        for i in range(1, n+1):
            # Update f_1 and f_2 values
            f_i_t_value[0] += self.df_1(f_i_t_value, f_eq_ij_t_value) * h
            f_i_t_value[1] += self.df_2(f_i_t_value, f_eq_ij_t_value) * h
            t += h
        
        # Returns t+1 value
        return f_i_t_value
    
    # Function to generate weights for trapezoid integration
    def weights(self):

        # Calculate stepsize to be used for weight
        h = (self.upper - self.lower)/self.step
        # Create array to use for weights
        arr = np.empty(self.step)
        arr.fill(h)
        # First and last weights are h/2
        arr[0] = h/2
        arr[self.step - 1] = h/2
        
        return arr


    # Function to numerically integrate in thress dimensions
    # Input: f as numpy array
    # Output: integrated function values
    def triple_integrate(self, f):
        
        # Integrate thrice
        result = np.array(f).dot(self.weight)
        result = result.dot(self.weight)
        result = result.dot(self.weight)

        return result
    

    # Function to calculate (p, m, e)
    # Input: f as numpy array
    # Output: [p, m, e]
    def p_m_e(self, f_i_t):

        #print('Calculating p, m, e')

        # Create storage
        pme = []

        # Iterate over species values
        for i in range(0, 2):

            # Momentum density integral function
            f_momden = [self.xx * f_i_t[i], self.yy * f_i_t[i], self.zz * f_i_t[i]]
            # Shape is (v_fx(mesh), v_fy(mesh), v_fz(mesh))

            # Energy density integral function
            v_sqr = self.xx**2 + self.yy**2 + self.zz**2
            f_enden = 0.5 * v_sqr * f_i_t[i]

            # Create f_extended for calculation
            f_extended = [f_i_t[i]] + f_momden + [f_enden]
            
            # Iterate over f_extended for integration
            p_m_e_value = [self.triple_integrate(f_extended[0]),
            self.triple_integrate(f_extended[1]), self.triple_integrate(f_extended[2]), 
            self.triple_integrate(f_extended[3]), self.triple_integrate(f_extended[4])]
            
            # Save results
            pme += [list(self.mass[i] * np.array(p_m_e_value))]

        #print(np.array(pme).shape)
        return pme

    # Function to calculate terms matrix
    # Input: f as numpy array
    # Output: (1, v_k, v_k_sq)
    def term(self, f_i_t):

        #print('Calculating coefficient matrix')
        
        # Create storage
        coef = []

        # Iterate over species values
        for i in range(0, 2):
            
            # Iterate over the 
            # Create the coefficent matrix for usage in Jacobian
            coef_shape = np.array(f_i_t[i]).shape
            coef_one = np.full(coef_shape, 1)
            coef_vk = [self.xx, self.yy, self.zz]
            v_sqr = self.xx**2 + self.yy**2 + self.zz**2
            coef_vk_sq = np.full(coef_shape, v_sqr)

            # Create coefficient extended for calculation
            coef_extended = [coef_one] + coef_vk + [coef_vk_sq]
            # Reshape the array
            coef_extended = list(np.array(coef_extended).transpose(1, 2, 3, 0))

            # Save the results 
            coef += [coef_extended]   

        return coef

    # Function to calculate Jacobian coefficients
    # Input: coefficient matrix
    # Output: Jacob_k matrix
    def jacob_k(self, coef):

        #print('Calculating Jacobian k coefficient matrix')

        # Create storage
        jacob = []

        # Iterate over species values
        for i in range(0, 2):

            # Extract the species specific coefficient matrix
            coef_i = coef[i]

            # Create storage
            jacob_i = []

            # Iterate over the axis of the matrix
            for x in range(0, len(coef_i)):
                y_result = []
                for y in range(0, len(coef_i)):
                    z_result = []
                    for z in range(0, len(coef_i)):

                        # Compute the transpose
                        result = np.multiply(np.array(coef_i[x][y][z]), np.array(coef_i[x][y][z]).reshape(5, 1))
                        z_result += [list(result)]
                    y_result += [z_result]
                
                jacob_i += [y_result]

            # Save the species result
            jacob += [jacob_i]
        
        #print(np.array(jacob).shape)
        return jacob


    # Function to calculate exponent
    # Input: f_i_t, abc is [[alpha, beta, gamma], [alpha, beta, gamma]]
    # Output: e ^ [a + b * v_k + c * v_k^2]
    def exponent(self, f_i_t, abc):

        #print('Calculating exponent for Jacobian')

        # Create storage
        expo = []

        # Iterate over species values
        for i in range(0, 2):
            
            # Get the values for this species
            abc_i = abc[i]
            # Create the alpha, beta, gamma matrix for usage in exponent
            coef_shape = np.array(f_i_t[i]).shape
            alpha = np.full(coef_shape, abc_i[0])
            beta_one = np.full(coef_shape, abc_i[1])
            beta_two = np.full(coef_shape, abc_i[2])
            beta_three = np.full(coef_shape, abc_i[3])
            gamma = np.full(coef_shape, abc_i[4])

            # Calculate the exponent
            beta_vk = beta_one * self.xx + beta_two * self.yy + beta_three * self.zz
            v_sqr = self.xx**2 + self.yy**2 + self.zz**2
            gamma_vk = gamma * v_sqr
            exponent = alpha + beta_vk + gamma_vk

            # Store the result
            expo_result = np.exp(exponent)
            expo += [expo_result]

        return expo
    

    # Function to calculate Jacob(x_k)
    # Input: jacob_k component, exponent component
    # Output: jacob_abc values
    def jacob_xk(self, jacob_k, exponent):

        #print('Calculating Jacobian xk')

        # Create storage
        jacob = []

        # Create the weight
        weight = self.weights()

        # Iterate over the species
        for i in range(0, 2):

            # Get the values for this species
            jacob_k_i = jacob_k[i]
            expo = exponent[i]

            # Storage for species specific jacob
            jacob_i = np.full((5,5), 0, dtype=float)

            # Iterate over the axis of the matrix
            for x in range(0, len(jacob_k_i)):
                for y in range(0, len(jacob_k_i)):
                    for z in range(0, len(jacob_k_i)):

                        # Compute the values
                        jacob_comp = np.array(jacob_k_i[x][y][z]) * expo[x][y][z]
                        # Multiply by the weights
                        jacob_comp *= weight[x] * weight[y] * weight[z]

                        # Save the result
                        jacob_i += jacob_comp

            jacob += [jacob_i]
                    
        return jacob

    # Function to calculate g(x_k)
    # Input: coefficient component, exponent component, pme matrix
    # Output: g_xk values
    def g_xk(self, coef, expo, pme):

        #print('Calculating g xk')

        # Create storage
        g_save = []

        # Create the weight
        weight = self.weights()

        # Iterate over the species
        for i in range(0, 2):

            # Get the values for this species
            coef_i = coef[i]
            expo_i = expo[i]
            pme_i = pme[i]
            
            # Storage for species specific g
            g_i = np.full((5), 0, dtype=float)

            # Iterate over the axis of the matrix
            for x in range(0, len(coef_i)):
                for y in range(0, len(coef_i)):
                    for z in range(0, len(coef_i)):

                        # Compute the values
                        g_comp = np.array(coef_i[x][y][z]) * expo_i[x][y][z]
                        # Multiply by the weights
                        g_comp *= weight[x] * weight[y] * weight[z]
                        
                        # Save the result
                        g_i += g_comp

            # Formula minus (p, m, 2E)
            g_i_result = g_i - (pme_i * np.array([1,1,1,1,2]))
            
            # Save the result in g
            g_save += [g_i_result]

        return g_save

    # Function to compute f_eq with netwon's method
    # Input: f_t value
    # Return: f_eq value
    def newton(self, f_i_t1_value, limit):

        # Calculate variables to compute jacobian and g_xk
        print('Computing pme')
        pme_value = self.p_m_e(f_i_t1_value)
        print('Computing coefficients')
        coef_value = self.term(f_i_t1_value)
        print('Computing jacob_k')
        jacob_k_value = self.jacob_k(coef_value)
        abc = [[1, -1, 0, 0, -1], [1, 1, 0, 0, -1]]
        print('Computing exponents')
        expo = self.exponent(f_i_t1_value, abc)

        # Compute jacobian and g_xk
        print('Computing j_xk')
        jacob_xk = self.jacob_xk(jacob_k_value, expo)
        print('Computing g_xk')
        g_xk_value = self.g_xk(coef_value, expo, pme_value)

        # Use newton process to calculate alpha, beta, gamma
        # For 2 species
        for i in range(0, 2):
            
            print('This is the ' + str(i) + 'th species')
            # Calculate the error term
            error_term = np.linalg.norm(g_xk_value[i])
            print('Current error size is now ' + str(round(error_term, 5)))

            counter = 0
            # While erorr term is large
            while error_term > limit:

                # j_xk inv multiply by g_xk
                j_inv_g = np.matmul(np.linalg.inv(jacob_xk[i]), g_xk_value[i])
                
                # Calculate next step x_k+1
                abc[i] -= j_inv_g
                # Use as input into g
                expo = self.exponent(f_i_t1_value, abc)
                jacob_xk = self.jacob_xk(jacob_k_value, expo)
                g_xk_value = self.g_xk(coef_value, expo, pme_value)
                
                # Compute error
                error_term = np.linalg.norm(g_xk_value[i])

                if counter % 10 == 0:
                    print('Iterating for the ' + str(counter) + 'th time')
                    print('Error size is now ' + str(round(error_term, 5)))
                    print(np.array(abc[i]).round(5))

                counter += 1

        # Once alpha beta gamma is computed, use the results to solve for f_eq
        f_eq = self.exponent(f_i_t1_value, abc)
            
        return f_eq


    '''
    Graphing Tools
    '''

    # Function to flatten by 1 dimension (remove z dimension)
    # Input: f as numpy array
    # Output: f with x, y only
    def flatten_z(self, f):
        
        # Shift z to the front for integration
        f = np.array(f).transpose(2, 0, 1)
        result = np.array(f).dot(self.weight)

        return result

    # Function to flatten by 1 dimension (remove y dimension)
    # Input: f as numpy array
    # Output: f with x only
    def flatten_y(self, f):
        
        # Shift z to the front for integration
        f = np.array(f).transpose(1, 0)
        result = np.array(f).dot(self.weight)

        return result
    
    # Function to retrieve the flattened results
    # Input: f as numpy array
    # Output: [f with x, y only], [f with x only]
    def flatten_all(self, f):

        xy_dim = self.flatten_z(f)
        x_dim = self.flatten_y(xy_dim)

        return xy_dim, x_dim


    # Function to flatten by 1 dimension (remove y dimension)
    # Input: f as numpy array (all species), time (timestamp) (float)
    # Output: f with x only
    def create_3d_plots(self, f, time):
        
        # for i in species
        for i in range(0, 2):

            # Get parameters
            f_i_t_xy, f_i_t_x = self.flatten_all(f[i])
            x_axis, y_axis = self.meshgrid_calc(self.lower, self.upper, 2)

            # Set directory
            cwd = os.getcwd()
            # Save file directory
            speciesLoc = 's' + str(i) + '_3d'
            saveFile = os.path.join(cwd, speciesLoc)
            # Save file location
            fig_name = 's' + str(i) + "_t" + str(round(float(time), 10)) + '_plot' + '.png'
            print('Creating ' + fig_name + ' plot')
            figLoc = os.path.join(saveFile, fig_name)

            # Plot the figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            surf = ax.contour3D(x_axis, y_axis, f_i_t_xy, 111, cmap=plt.cm.viridis)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('f')
            ax.set_zlim(0, 0.3)
            ax.view_init(30, 45) # Rotation
            fig.set_size_inches(12, 8, forward=True)
            ax.dist = 12
            plt.savefig(figLoc)
            plt.close()

    # Function to flatten by 1 dimension (remove y dimension)
    # Input: f as numpy array (all species), time (timestamp) (float)
    # Output: f with x only
    def create_2d_plots(self, f, time):
        
        # Set directory
        cwd = os.getcwd()
        # Save file directory
        speciesLoc = 's01_2d'
        saveFile = os.path.join(cwd, speciesLoc)
        # Save file location
        fig_name = "t" + str(round(float(time), 10)) + '_plot' + '.png'
        print('Creating ' + fig_name + ' plot')
        figLoc = os.path.join(saveFile, fig_name)
        
        # Get parameters
        f_0_t_xy, f_0_t_x = self.flatten_all(f[0])
        f_1_t_xy, f_1_t_x = self.flatten_all(f[1])
        x_axis = self.meshgrid_calc(self.lower, self.upper, 1)

        # Create dataframe to store data to plot
        d = {'x': x_axis, 'f_0': f_0_t_x, 'f_1': f_1_t_x}
        df = pd.DataFrame(data = d)

        # Plot the figure
        sns.set_style("whitegrid")
        ax = sns.lineplot(data=df, x='x', y='f_0', label = 'f_0')
        sns.lineplot(data=df, x='x', y='f_1', label = 'f_1')
        ax.set(xlabel='x', ylabel='f')
        ax.legend()
        ax.set(xlim=(min(x_axis-1),max(x_axis+1)),ylim=(0,1))
        plt.savefig(figLoc)
        plt.close()


    '''
    Main section to run the script
    '''

    def run(self):

        # To track run time
        start = timeit.timeit()

        # Calculate the initial values
        f_i_t_value = self.f_i_0()
        f_eq_ij_t_value = self.f_eq_ij_v_0()     
        # Save the data       
        self.create_2d_plots(f_i_t_value, 0)
        self.create_3d_plots(f_i_t_value, 0)

        # Create array to run calculations across time t
        timeMesh = np.linspace(0, 1.5, 100)

        # Track time
        end = timeit.timeit()
        timeelapsed = end - start
        print('Time Elapsed: ' + str(timeelapsed))

        # Iterate over time
        for i in range(1, len(timeMesh)):
            
            # Get the time range for this step of calculation
            time_range = [timeMesh[i-1], timeMesh[i]]
            
            # Calculate next f_i value
            f_i_t1_value = self.euler(f_i_t_value,  f_eq_ij_t_value, time_range, 0.0001)
            # Save the data
            self.create_2d_plots(f_i_t1_value, timeMesh[i])
            self.create_3d_plots(f_i_t1_value, timeMesh[i])

            # Track time
            end = timeit.timeit()
            timeelapsed = end - start
            print('Time Elapsed: ' + str(timeelapsed))

            # Calculate next f_eq value
            f_eq_ij_t_value = self.newton(f_i_t1_value, 0.1)


        # Calculate final f_i value
        f_i_t1_value = self.euler(f_i_t_value,  f_eq_ij_t_value, time_range, 0.0001)

#%%

'''
Running the script with parameters
'''

BGK_Calc = BGK(20, -5, 5)
#BGK_Calc = BGK(20, -5, 5)
BGK_Calc.run()


# %%
