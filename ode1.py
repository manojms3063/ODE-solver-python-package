import numpy as np
import sympy as smp
from sympy import symbols,diff
import math
import numpy as np
import sympy as smp
from sympy import symbols

def euler(f):
    print("Euler's Method to Solve 1st Order ODE Numerically.")
    
    # User inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    r = float(input("Enter the value where the value to be found (r) = "))

    # Define symbolic variables
    x_sym, y_sym = symbols('x y')

    # Convert symbolic function to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)

    # Number of steps
    n = abs(int((b - a) / h))

    # Initialize arrays
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initial conditions
    x[0] = x0
    y[0] = y0

    # Euler's Method Loop
    for i in range(n):
        y[i + 1] = y[i] + h * fxy(x[i], y[i])
        x[i + 1] = x[i] + h

    # Output results
    print('x =', x)
    print('y =', y)

    # Find y(r)
    m = np.where(np.isclose(x, r))
    if m[0].size > 0:
        print(f"y({r}) = {y[m][0]}")
    else:
        print(f"Value at x = {r} is not explicitly computed.")




def eulermodi(f):
    print("Modified Euler's Method to Solve 1st Order ODE Numerically.")
    
    # Define symbolic variables
    x_sym, y_sym = symbols('x y')

    # Convert symbolic function to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)  

    # User Inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    r = float(input("Enter the value where the value to be found (r) = "))

    # Number of steps
    n = int((b - a) / h)

    # Initialize arrays
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initial conditions
    x[0] = x0
    y[0] = y0

    # Step 1: Euler's Predictor Step
    for i in range(n):
        y[i + 1] = y[i] + h * fxy(x[i], y[i])
        x[i + 1] = x[i] + h

    print(f"Predictor values (Euler's step): y* =", y)

    # Step 2: Corrector Step (Modified Euler Method)
    for j in range(n):
        y[j + 1] = y[j] + (h / 2) * (fxy(x[j], y[j]) + fxy(x[j + 1], y[j + 1]))

    # Print Results
    print("x =", x)
    print("y =", y)

    # Find y(r)
    m = np.where(np.isclose(x, r))
    if m[0].size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")


def rk2(f):
    print("Runge-Kutta 2nd Order Method to Solve 1st Order ODE Numerically.")

    # Define symbolic variables
    x_sym, y_sym = symbols('x y')

    # Convert symbolic function to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)  

    # User Inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    r = float(input("Enter the value where the value to be found (r) = "))

    # Number of steps
    n = int((b - a) / h)

    # Initialize arrays
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initial conditions
    x[0] = x0
    y[0] = y0

    # Runge-Kutta 2nd order method
    for i in range(n):
        k1 = h * fxy(x[i], y[i])
        k2 = h * fxy(x[i] + h, y[i] + k1)
        
        y[i + 1] = y[i] + (1 / 2) * (k1 + k2)
        x[i + 1] = x[i] + h

    # Print results
    print("x =", x)
    print("y =", y)

    # Find y(r)
    m = np.where(np.isclose(x, r))
    if m[0].size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")
        
        

def rk4(f):
    print("Runge-Kutta 4th Order Method to Solve 1st Order ODE Numerically.\n")

    # Define symbolic variables
    x_sym, y_sym = symbols('x y')

    # Convert symbolic function to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)  

    # User Inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    r = float(input("Enter the value where the value to be found (r) = "))

    # Number of steps
    n = int((b - a) / h)
    print(f"Number of steps: {n}")

    # Initialize arrays
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initial conditions
    x[0] = x0
    y[0] = y0

    # Runge-Kutta 4th Order Method
    for i in range(n):
        k1 = h * fxy(x[i], y[i])
        k2 = h * fxy(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * fxy(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * fxy(x[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i + 1] = x[i] + h

    # Print results
    print("x =", x)
    print("y =", y)

    # Find y(r)
    m = np.where(np.isclose(x, r))
    if m[0].size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")



def taylor_series(f, g):
    print("Taylor Series Method to Solve 1st Order ODE Numerically.\n")

    # Define symbolic variables
    x_sym, y_sym = symbols('x y')

    # Convert symbolic expressions to numerical functions
    dydx = f(x_sym, y_sym)
    d2ydx2 = g(x_sym, y_sym)

    fxy = smp.lambdify([x_sym, y_sym], dydx)
    dfxy = smp.lambdify([x_sym, y_sym], d2ydx2)

    # User Inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    r = float(input("Enter the value where the value to be found (r) = "))

    # Number of steps
    n = int((b - a) / h)
    print(f"Number of steps: {n}")

    # Initialize arrays
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initial conditions
    x[0] = x0
    y[0] = y0

    # Taylor Series Method
    for i in range(n):
        y[i+1] = y[i] + h * fxy(x[i], y[i]) + (h**2 / 2) * dfxy(x[i], y[i])
        x[i+1] = x[i] + h

    # Print results
    print("x =", x)
    print("y =", y)

    # Find y(r)
    m = np.where(np.isclose(x, r))
    if m[0].size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")
        
        
        
def milne_pc(f):
    print("Milne Predictor-Corrector Method to Solve 1st Order ODE Numerically.\n")

    # Define symbolic variables
    x_sym, y_sym = symbols("x y")

    # Convert symbolic expression to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)

    # User Inputs
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    y1 = float(input("Enter y1 = "))
    y2 = float(input("Enter y2 = "))
    y3 = float(input("Enter y3 = "))
    h = float(input("Enter Step Size (h) = "))
    r = float(input("Enter the value where y(r) is needed (r) = "))

    # Number of steps
    n = int((b - a) / h)
    print(f"Number of steps: {n}")

    # Initialize arrays
    y = np.array([y0, y1, y2, y3])
    x = np.array([x0 + i * h for i in range(4)])

    print("x =", x)
    print("y =", y)

    # Compute derivatives
    dy = np.array([fxy(x[i], y[i]) for i in range(4)])
    print("dy =", dy)

    # **Milne Predictor Formula**:
    y4_pred = y[0] + (4 * h / 3) * (2 * dy[1] - dy[2] + 2 * dy[3])
    print("Predicted y4 =", y4_pred)

    # Compute dy4_pred
    dy4_pred = fxy(x[3] + h, y4_pred)

    # **Milne Corrector Formula**:
    y4_corr = y[2] + (h / 3) * (dy[2] + 4 * dy[3] + dy4_pred)
    print("Corrected y4 =", y4_corr)

    # Find y(r)
    x = np.append(x, x[3] + h)  # Append x4
    y = np.append(y, y4_corr)  # Append y4_corrected

    m = np.where(np.isclose(x, r))[0]
    if m.size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")



def abpc(f):
    print("Adams-Bashforth Predictor-Corrector Method to Solve 1st Order ODE Numerically.\n")

    # Define symbolic variables
    x_sym, y_sym = symbols("x y")

    # Convert symbolic expression to numerical function
    dydx = f(x_sym, y_sym)
    fxy = smp.lambdify([x_sym, y_sym], dydx)

    # User Inputs
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    y1 = float(input("Enter y1 = "))
    y2 = float(input("Enter y2 = "))
    y3 = float(input("Enter y3 = "))
    h = float(input("Enter Step Size (h) = "))
    r = float(input("Enter the value where y(r) is needed (r) = "))

    # Number of steps
    n = int((b - a) / h)
    print(f"Number of steps: {n}")

    # Initialize arrays
    y = np.array([y0, y1, y2, y3])
    x = np.array([x0 + i * h for i in range(4)])

    print("x =", x)
    print("y =", y)

    # Compute derivatives
    dy = np.array([fxy(x[i], y[i]) for i in range(4)])
    print("dy =", dy)

    # **Adams-Bashforth Predictor Formula**:
    y4_pred = y[3] + (h / 24) * (55 * dy[3] - 59 * dy[2] + 37 * dy[1] - 9 * dy[0])
    print("Predicted y4 =", y4_pred)

    # Compute dy4_pred
    dy4_pred = fxy(x[3] + h, y4_pred)

    # **Adams-Moulton Corrector Formula**:
    y4_corr = y[3] + (h / 24) * (9 * dy4_pred + 19 * dy[3] - 5 * dy[2] + dy[1])
    print("Corrected y4 =", y4_corr)

    # Append the new value
    x = np.append(x, x[3] + h)  # Append x4
    y = np.append(y, y4_corr)  # Append y4_corrected

    # Find y(r)
    m = np.where(np.isclose(x, r))[0]
    if m.size > 0:
        print(f"y({r}) =", y[m][0])
    else:
        print(f"Value at x = {r} is not explicitly computed.")
    
    
            
            
            
                
                
