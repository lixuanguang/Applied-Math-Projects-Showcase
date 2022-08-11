#%%
'''

MA5232 Part 3 - Assignment
Q5 - Movie Lens

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

####################### Solve Q5 ########################
# Class function to solve the question
class movlens:

    # Intialise object
    def __init__(self):
        self.trainingDf, self.testDf = self.splitDf()

    # Load data
    # Data directory assumed to be at:
    # Parent folder
    #     - movlens.py
    #     - data (Child folder)
    #           - data.csv
    def loadData(self):

        # Set directory
        cwd = os.getcwd()
        # Data file directory
        dataFile = os.path.join(cwd, 'data')
        dataFile = os.path.join(dataFile, 'data.csv')
        df = pd.read_csv(dataFile, header = None)
        df.columns = ['user_id', 'movie_id', 'rating']

        return df

    # Create training and testing dataset
    # Input: none
    # Output: training and test dataset
    def splitDf(self):

        # Load the dataset
        df = self.loadData()
        # Split the dataset
        train = df.sample(frac = 0.9)
        test = df.drop(train.index)

        return train, test

    # Function for baseline estimator
    def baseline(self):

        # Compute training df average ratings
        train = self.trainingDf[['movie_id', 'rating']]
        train_base = train.groupby('movie_id').mean().reset_index()
        train_base.columns = ['movie_id', 'mean_rating']
        # Left join on test dataset
        test_append = self.testDf.merge(train_base, on = 'movie_id',  how = 'left')
        test_append['squared_diff'] = (test_append['rating'] - test_append['mean_rating']) ** 2
        # Calculate squared error loss, mean squared error
        sqloss = test_append['squared_diff'].sum()
        MSE = (1/len(test_append)) * sqloss

        return sqloss, MSE

    # Function to plot and save the figures
    # Input: error df, r count
    def save_fig(self, df, fig_name):

        ### Set save file location
        # Set directory
        cwd = os.getcwd()
        # Save file directory
        fileLoc = 'result'
        saveFile = os.path.join(cwd, fileLoc)
        # Save file location
        print('Creating ' + fig_name + ' plot')
        figLoc = os.path.join(saveFile, fig_name)

        # Plot the figure
        fig,ax = plt.subplots()
        for r in range(1, 21):
            colName = "r_" + str(r) + "_val"
            ax.plot(df['Iteration'], df[colName], label = colName)

        ax.set_xlabel("Iterrations")
        ax.set_ylabel("Squared Error")
        ax.legend(loc='best')
        plt.savefig(figLoc)
        plt.close()

    # Function to plot and save the figures
    # Input: error df with different r, type = str
    def save_fig_2(self, df, typeG):

        ### Set save file location
        # Set directory
        cwd = os.getcwd()
        # Save file directory
        fileLoc = 'result'
        saveFile = os.path.join(cwd, fileLoc)
        # Save file location
        fig_name = 'q5v_' + str(typeG) + '.png'
        print('Creating ' + fig_name + ' plot')
        figLoc = os.path.join(saveFile, fig_name)

        # Set the axis largest and smallest values
        ymax = df[typeG].max()
        ymin = df[typeG].min()

        # Plot the figure
        fig = plt.figure()
        ax = sns.lineplot(data = df, x = 'r_value', y = typeG)
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([1, 20])
        plt.savefig(figLoc)
        plt.close()

    
    # Function for matrix completion algorithm
    # Input: r values
    def matrixComp(self, r):

        print("Computing for r value = " + str(r))

        # 943 user id, 1682 movie id
        X_mat = np.zeros((944, 1683), dtype = float) # User axis on y-axis, movie on x-axis

        training = self.trainingDf.to_numpy()
        # Populate the X matrix with training df
        i = training[:, 0]
        j = training[:, 1]
        v = training[:, 2]
        X_mat[i, j] = v

        # Keep original copy for calculating error
        X_original = np.copy(X_mat)

        # Compute the SVD of matrix
        U_mat, S_mat, V_mat_T = np.linalg.svd(X_mat, full_matrices = False)

        # Create df to store error values
        colName = "r_" + str(r) + "_val"
        errorDf = pd.DataFrame(columns = ['Iteration', colName])
        
        # Iterate for 200 times
        for iter in range(0, 200):

            # If its not a first ieration, then we need to recompute
            if iter != 0:

                # Populate the X matrix with training df
                for index, row in self.trainingDf.iterrows():

                    u_id = row['user_id']
                    m_id = row['movie_id']
                    rat = row['rating']

                    X_mat[u_id][m_id] = rat

                # Compute the SVD of matrix
                U_mat, S_mat, V_mat_T = np.linalg.svd(X_mat, full_matrices=False)
                
            # Calculate S_new
            S_mat[:r] = 0

            # Update X_new
            X_new = U_mat @ np.diag(S_mat) @ V_mat_T

            # Calculate the Frobenius norm
            error = np.linalg.norm(X_mat - X_new, 'fro')

            # Save the error output
            errorDf = errorDf.append({'Iteration': iter+1, colName: error}, ignore_index=True)
            
            # Then updated X
            X_mat = X_new
        
            # To keep track of iterations
            if iter % 10 == 0:
                print("Error is now " + str(error))

        # Return the result
        return X_original, X_mat, errorDf

    # Function to compute squared error loss
    def sqerlos(self, X_mat):

        # 943 user id, 1682 movie id
        test_mat = np.zeros((944, 1683), dtype = float) # User axis on y-axis, movie on x-axis

        test = self.testDf.to_numpy()
        # Populate the test matrix with testing df
        i = test[:, 0]
        j = test[:, 1]
        v = test[:, 2]
        test_mat[i, j] = v

        # Get the corresponding predicted values
        pred_mat = np.zeros((944, 1683), dtype = float) # User axis on y-axis, movie on x-axis
        pred_mat[i, j] = X_mat[i, j]

        zeroes = 0
        for arr in test_mat:
            zeroes += np.count_nonzero(arr==0)
        
        print(zeroes)

        zeroes = 0
        for arr in pred_mat:
            zeroes += np.count_nonzero(arr==0)
        
        print(zeroes)

        # Calculate the squared error loss and mean squared error
        sqloss = np.sum(np.square(X_mat - pred_mat))
        MSE = (1/len(self.testDf)) * sqloss

        return sqloss, MSE
    
    # Function to run the program
    def run(self):

        sqloss_baseline, MSE_baseline = self.baseline()

        # Create df to store error values
        errorDf = pd.DataFrame(columns = ['r_value', 'SqLoss', 'MSE'])
        iterErrDf = pd.DataFrame()

        # Iterate over the values
        for r in range(1, 21):

            X_original, X_mat, iterErrDf_res = self.matrixComp(r)

            if iterErrDf.empty:
                iterErrDf = iterErrDf_res
            else:
                iterErrDf = iterErrDf.merge(iterErrDf_res, on = 'Iteration', how = 'left')

            X_mat_clip = np.copy(X_mat)
            X_mat_clip[X_mat_clip[:, :] > 5] = 5
            X_mat_clip[X_mat_clip[:, :] < 1] = 1

            sqloss, MSE = self.sqerlos(X_mat_clip)
            # Save the error output
            errorDf = errorDf.append({'r_value': r, 'SqLoss': sqloss, 'MSE': MSE}, ignore_index=True)
        
        # Plot the charts
        self.save_fig(iterErrDf, 'q5iv_noclip.png')
        self.save_fig_2(errorDf, 'SqLoss')
        self.save_fig_2(errorDf, 'MSE')

        return errorDf, [sqloss_baseline, MSE_baseline]

#%%

movlensCalc = movlens()
errorDf, other_error = movlensCalc.run()


# %%
# %%
