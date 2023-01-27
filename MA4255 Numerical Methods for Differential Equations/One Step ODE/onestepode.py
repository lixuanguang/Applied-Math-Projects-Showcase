
import numpy as np
import math
import matplotlib.pyplot as plt

class onestepode:

    def __init__(self):
        self.y_0 = 1
        self.x_0 = 0
        self.x_f = 10
        self.h_val = 1/10

    # Inputs: y' = func, initial value, x start range, x end range, step size
    def explicit_euler(self, func, y_0=1, x_0=0, x_f=10, h_val=1/10):

        t = np.arange(x_0, x_f+h_val, h_val) # Create array of equally spaced values
        y = np.zeros(t.shape) # Initialise zeros of y first
        y[0] = y_0 # IVP value

        # Time step
        for i in range(1, len(t)):
            y[i] = y[i-1] + h_val*func(t[i-1], y[i-1]) # y_{n+1} = y_n + h * f(x_n, y_n)
        
        print('The solution of Explicit Euler is as follows')
        print(t)
        print(y)

        # plotting the points 
        plt.plot(t, y)
        
        # naming the x axis
        plt.xlabel('X - Axis')
        # naming the y axis
        plt.ylabel('Y - Axis')
        
        # giving a title to graph
        plt.title('Explicit Euler')
        plt.savefig('Explicit Euler.png', bbox_inches='tight')
        
        # function to show the plot
        plt.show()

        return t, y


    # Inputs: y' = func, dfdy function, initial value, x start range, x end range, step size
    def implicit_euler(self, func, dfdy, y_0=1, x_0=0, x_f=10, h_val=1/10, tol=1e-6, max_iter=1000):

        t = np.arange(x_0, x_f+h_val, h_val) # Create array of equally spaced values
        y = np.zeros(t.shape) # Initialise zeros of y first
        n = int((x_f - x_0)/h_val) # Number of steps
        y[0] = y_0 # IVP value

        # Time Stepping
        for x in range(n):

            # Initial guess for y_next
            y_next = y[x] + h_val * func(t[x], y[x])

            for i in range(max_iter):
                # Newton-Raphson method: y_next = y_next - f(y_next)/f'(y_next)
                f_y_next = y_next - y[x] - h_val * func(t[x] + h_val, y_next)
                f_prime_y_next = 1 - h_val * dfdy(t[x] + h_val, y_next)
                y_next = y_next - f_y_next / f_prime_y_next
                if abs(f_y_next) < tol:
                    break
            t[x+1] = t[x] + h_val
            y[x+1] = y_next

        print('The solution of Implicit Euler is as follows')
        print(t)
        print(y)

        # plotting the points 
        plt.plot(t, y)
        
        # naming the x axis
        plt.xlabel('X - Axis')
        # naming the y axis
        plt.ylabel('Y - Axis')
        
        # giving a title to graph
        plt.title('Implicit Euler')
        plt.savefig('Implicit Euler.png', bbox_inches='tight')
        
        # function to show the plot
        plt.show()

        return t,y


    # Inputs: y' = func, initial value, x start range, x end range, step size
    def implicit_midpoint(self, func, y_0=1, x_0=0, x_f=10, h_val=1/10, tol=1e-6, max_iter=1000):

        t = np.arange(x_0, x_f+h_val, h_val) # Create array of equally spaced values
        y = np.zeros(t.shape) # Initialise zeros of y first
        n = int((x_f - x_0)/h_val) # Number of steps
        y[0] = y_0 # IVP value

        # Time Stepping
        for x in range(n):

            # Initial guess for y_next
            y_next = y[x] + h_val * func(t[x] + h_val/2, y[x] + h_val/2 * func(t[x], y[x]))

            t[x+1] = t[x] + h_val
            y[x+1] = y_next

        print('The solution of Implicit Midpoint Rule is as follows')
        print(t)
        print(y)

        # plotting the points 
        plt.plot(t, y)
        
        # naming the x axis
        plt.xlabel('X - Axis')
        # naming the y axis
        plt.ylabel('Y - Axis')
        
        # giving a title to graph
        plt.title('Implicit Midpoint Rule')
        plt.savefig('Implicit Midpoint Rule.png', bbox_inches='tight')
        
        # function to show the plot
        plt.show()

        return t,y


if __name__ == '__main__':

    def f(x,y):
        return math.sin(x**2)*y

    def dfdy(t,y):
        return np.sin(t**2)
    
    oderun = onestepode()
    expeul_t, expeul_y = oderun.explicit_euler(f)
    impeul_t, impeul_y = oderun.implicit_euler(f, dfdy)
    impmid_t, impmid_y = oderun.implicit_midpoint(f)

    t = np.arange(0, 10.1, 0.1)

    # Plotting both the curves simultaneously
    plt.plot(t, expeul_y, color='r', label='Explicit Euler')
    plt.plot(t, impeul_y, color='g', label='Implicit Euler')
    plt.plot(t, impmid_y, color='b', label='Implicit Midpoint Rule')
    
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Comparison of Values")
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # giving a title to graph
    plt.title('One Step Methods')
    plt.savefig('One Step Methods.png', bbox_inches='tight')
    
    # To load the display window
    plt.show()