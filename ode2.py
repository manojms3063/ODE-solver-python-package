import numpy as np
import sympy as smp
from sympy import symbols,diff
import math


def runge_kutta_4(f, g):
    print("\nRunge-Kutta 4th Order Method for Solving System of ODEs\n")

    # Define symbolic variables
    x_sym, y_sym, z_sym = symbols("x y z")

    # Convert symbolic expressions to numerical functions
    dydx = f(x_sym, y_sym, z_sym)
    dzdx = g(x_sym, y_sym, z_sym)
    fxy = smp.lambdify([x_sym, y_sym, z_sym], dydx)
    gxyz = smp.lambdify([x_sym, y_sym, z_sym], dzdx)

    # User Inputs
    h = float(input("Enter Step Size (h) = "))
    a = float(input("Enter start point (a) = "))
    b = float(input("Enter end point (b) = "))
    x0 = float(input("Enter initial value (x0) = "))
    y0 = float(input("Enter initial value (y0) = "))
    z0 = float(input("Enter initial value (z0) = "))
    r = float(input("Enter the value where y(r) and z(r) are needed (r) = "))

    # Number of steps
    n = int((b - a) / h)
    print(f"Number of steps: {n}")

    # Initialize arrays
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    z = np.empty(n + 1)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    # Runge-Kutta iterations
    for i in range(n):
        k1 = h * fxy(x[i], y[i], z[i])
        l1 = h * gxyz(x[i], y[i], z[i])

        k2 = h * fxy(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        l2 = h * gxyz(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)

        k3 = h * fxy(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        l3 = h * gxyz(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)

        k4 = h * fxy(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * gxyz(x[i] + h, y[i] + k3, z[i] + l3)

        y[i+1] = y[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        z[i+1] = z[i] + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
        x[i+1] = x[i] + h

    # Print results
    print("\nComputed Values:")
    print("x =", x)
    print("y =", y)
    print("z =", z)

    # Find y(r) and z(r)
    m = np.where(np.isclose(x, r))[0]
    if m.size > 0:
        print(f"\nApproximate y({r}) = {y[m][0]}")
        print(f"Approximate z({r}) = {z[m][0]}")
    else:
        print(f"\nValue at x = {r} is not explicitly computed.")

    m = np.where(x==r)[0]
    print(f"y({b})=",y[m])
    print(f"z({b})=",z[m])
    
