#%%
'''

MA5232 Part 3 - Assignment
Q4 - Matrix Completion

Creator: Li Xuanguang
Student ID: A0154735B

'''
#%%

####################### IMPORT MODULES ########################
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import math
import cvxpy as cp
import os
import seaborn as sns

#%%

####################### Solve Q4 ########################
# Class function to solve the question
class matcom:

    # Intialise object with n, m
    # Input the dimensions of matrix Z
    def __init__(self, m, n):
        self.m = m
        self.n = n

    # Function to initialise the random matrix Z
    # Inputs: none
    # Outputs: random matrix Z with rank r = 2
    def createMatrixZ(self):

        print("Creating matrix Z")

        # Specify the parameters
        mu, sigma = 0, 1 # mean and standard deviation

        # Generate matrix U, V first
        U_mat = np.random.normal(mu, sigma, (self.m, 2))
        V_mat = np.random.normal(mu, sigma, (self.n, 2))

        # Create matrix Z
        Z_mat = np.matmul(U_mat, V_mat.T)

        return Z_mat

    # Function to initialise the random matrix X
    # Inputs: matrix Z, k value
    # Outputs: random matrix X with randomly chosen locations = k
    def createMatrixX(self, Z_mat, k):

        # Create data to store the results - initilise as empty
        data = np.empty((self.m, self.n))
        data[:] = np.nan

        # Create omega
        array_temp = np.arange(self.m * self.n) # Make evenly spaced intervals
        perm_arr = np.random.permutation(array_temp) # Then do a permutation
        chosen_loc = np.sort(perm_arr[0:k]) # Choose k samples from the array
        omega = np.unravel_index(chosen_loc, (self.m, self.n)) # Conver to coordinate
        data[omega] = Z_mat[omega] # Set the values

        # Get complement
        chosen_loc_com = np.sort(perm_arr[k:]) # Choose k samples from the array
        omega_com = np.unravel_index(chosen_loc_com, (self.m, self.n))

        return data, omega, omega_com

    # Function to run the minimisation problem
    # Inputs: Z_mat, omega
    # Output: solution value, x_var solved
    def minimise(self, Z_mat, omega):

        # Set the minimsation variable
        X_var = cp.Variable((self.m, self.n))
        # Set the objective function and constraints
        obj_func = cp.norm(X_var, 'nuc')
        constraints = [X_var[omega] == Z_mat[omega]]

        # Solve the problem
        problem = cp.Problem(cp.Minimize(obj_func), constraints)
        problem.solve(verbose = True)

        # Return the values
        return problem.value, X_var.value

    # Function to calculate the error
    # Inputs: calculated X matrix, generated Z-matrix, omegacomplement
    def errorCal(self, X_var, Z_mat, omega_c):

        # Square the difference of the matrix entries
        sq_ele = np.square(X_var[omega_c] - Z_mat[omega_c])
        # Then sum the entries
        sq_sum_ele = np.sum(sq_ele)

        # Get magnitude of omega_c
        omega_c_mag = len(omega_c[0])
        
        # Compute MSE
        MSE = sq_sum_ele / omega_c_mag

        return MSE

    # Function to run the program
    def run(self):

        # Create the matrix Z
        Z_mat = self.createMatrixZ()

        # Create df to store MSE values
        df = pd.DataFrame(columns = ['K_Value', 'MSE'])

        # Iterate over range of values
        for i in range (1, 31):

            # Run the program
            k = i * 100
            print("Running the program for k =" + str(k) + " value")
            X_mat, omega, omega_com = self.createMatrixX(Z_mat, k)
            opt_value, X_value = self.minimise(Z_mat, omega)
            error = self.errorCal(X_value, Z_mat, omega_com)

            # Save the error output
            df = df.append({'K_Value': k, 'MSE': error}, ignore_index=True)
        
        ### Set save file location
        # Set directory
        cwd = os.getcwd()
        # Save file directory
        fileLoc = 'result'
        saveFile = os.path.join(cwd, fileLoc)
        # Save file location
        fig_name = 'q4iv' + '.png'
        print('Creating ' + fig_name + ' plot')
        figLoc = os.path.join(saveFile, fig_name)

        # Plot the figure
        fig = plt.figure()
        sns.lineplot(data=df, x="K_Value", y="MSE")
        plt.savefig(figLoc)
        plt.close()

        return df

#%%

matcomCalc = matcom(100, 100)
result = matcomCalc.run()
# %%