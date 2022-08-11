#%%
'''

MA5232 Part 3 - Assignment
Q3 - Compressed sensing

Creator: Li Xuanguang
Student ID: A0154735B

'''
#%%

####################### IMPORT MODULES ########################
from re import A
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import math
import cvxpy as cp
import os

#%%

####################### Solve Q3 ########################
# Class function to solve the question
class comsens:

    # Intialise object with n, m, s
    # Input the max possible value
    # Program will iterate over every combination of m and s till max
    def __init__(self, n, m, s):
        self.n = n
        self.m = m
        self.s = s

    # Function to generate random m by n matrix A, x sparse
    # Inputs: m rows, s sparsity
    # Output: matrix A with each entry i.i.d. rv from norm distribution
    #         x sparse vector
    #         c a vector with only 1 entries
    def generator(self, m, s):

        # Specify the parameters
        mu, sigma = 0, 0.1 # mean and standard deviation

        # Create the matrix
        A_mat = np.random.normal(mu, sigma, (m, self.n))

        # Create the x* random sparse vector
        x_s = np.full((self.n), 0)
        while s != 0:
            # Choose random posiion to change to 1
            position = np.random.randint(self.n)
            # If position is not 0, change to 1, then decrease s
            if x_s[position] != 1:
                x_s[position] = 1
                s -= 1

        # Create the c vector with only 1 entires
        c_vec = np.full((self.n), 1)
 
        return A_mat, x_s, c_vec

    
    # Function to create the LP and solve
    # Inputs: A matrix, x sparse vector, c with only 1 entries vector
    # Outputs: Solution value, x array value
    def lpSolve(self, A_mat, x_s, c_vec):

        # Calculate y vector
        y = np.matmul(A_mat, x_s)
        # Define and solve the CVXPY problem.
        x = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(c_vec.T@x), [A_mat@x == y, x >= 0])
        solution = prob.solve()

        return solution, x.value

    # Function to determine success
    # Inputs: x solution, x random sparse vector
    def success(self, x_sol, x_s):

        # Numerator
        diff_vec = x_sol - x_s
        num = np.linalg.norm(diff_vec)
        # Denominator
        denom = np.linalg.norm(x_s)
        # Value
        val = num / denom

        # If less than value, declare success
        if val <= 10 ** (-4):
            return 1
        else:
            return 0


    # Function to run the program
    def run(self):

        # Create data to store the results
        data = np.full((self.s, self.m), 0)

        # Repeat experiment 10 times
        for i in range(0, 10):

            print("Running  " + str(i+1) + "th iteration")

            # Iterate for diff m and s values
            for m_val in range(1, self.m + 1):
                for s_val in range(1, self.s + 1):
                    
                    # Run the LP program
                    A_mat, x_s, c_vec = self.generator(m_val, s_val)
                    solution, x_val = self.lpSolve(A_mat, x_s, c_vec)
                    successResult = self.success(x_val, x_s)

                    # Save result in array
                    data[s_val-1][m_val-1] += successResult
        
        # Finally find average number of successes by dividing
        data = data/10
        
        ### Set save file location
        # Set directory
        cwd = os.getcwd()
        # Save file directory
        fileLoc = 'result'
        saveFile = os.path.join(cwd, fileLoc)
        # Save file location
        fig_name = 'q3iv' + '.png'
        print('Creating ' + fig_name + ' plot')
        figLoc = os.path.join(saveFile, fig_name)

        # Plot the figure
        fig = plt.figure()
        plt.imshow(data, cmap = 'gray', origin='lower')
        plt.savefig(figLoc)
        plt.close()

        return data

#%%

comsensCalc = comsens(50, 50, 50)
result = comsensCalc.run()

# %%
print(result[0])

# %%
