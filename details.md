# ODE-solver-python-package
# Numerical Methods for Solving Ordinary Differential Equations (ODEs) in Python

This repository contains Python implementations of various numerical methods for solving **first-order** and **second-order** ordinary differential equations (ODEs). These methods are widely used in numerical analysis and applied mathematics for solving differential equations that cannot be solved analytically.

## Implemented Methods

### 1st Order ODE Methods
1. **Euler's Method**  
   - A simple numerical method that approximates the solution of a first-order ODE using a finite difference approach.
   - Formula:
    $y_{n+1} = y_n + h f(x_n, y_n)$
   - **Usage in Python:**
     ```python
     def euler(f):
         ...  # Implementation of Euler's method
     ```

2. **Modified Euler's Method (Heun's Method)**  
   - An improved version of Euler's method using an iterative correction.
   - Formula:
     $y^* = y_n + h f(x_n, y_n)$
     $y_{n+1} = y_n + \frac{h}{2} [ f(x_n, y_n) + f(x_{n+1}, y^*) ]$
   - **Usage in Python:**
     ```python
     def modified_euler(f):
         ...  # Implementation of Modified Euler's method
     ```

3. **Runge-Kutta 2nd Order Method (RK2)**  
   - A two-stage explicit Runge-Kutta method.
   - Formula:
     $k_1 = h f(x_n, y_n)$
     $k_2 = h f(x_n + h, y_n + k_1)$
     $y_{n+1} = y_n + \frac{1}{2} (k_1 + k_2)$
   - **Usage in Python:**
     ```python
     def rk2(f):
         ...  # Implementation of RK2 method
     ```

4. **Runge-Kutta 4th Order Method (RK4)**  
   - A widely used high-accuracy method for solving ODEs.
   - Formula:
     $k_1 = h f(x_n, y_n)$
     $k_2 = h f(x_n + \frac{h}{2}, y_n + \frac{k_1}{2})$
     $k_3 = h f(x_n + \frac{h}{2}, y_n + \frac{k_2}{2})$
     $k_4 = h f(x_n + h, y_n + k_3)$
     $y_{n+1} = y_n + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)$
   - **Usage in Python:**
     ```python
     def rk4(f):
         ...  # Implementation of RK4 method
     ```

5. **Taylor Series Method**  
   - Uses the Taylor series expansion to approximate the solution.
   - Formula:
     $y_{n+1} = y_n + h f(x_n, y_n) + \frac{h^2}{2} f'(x_n, y_n)$
   - **Usage in Python:**
     ```python
     def taylor(f, g):
         ...  # Implementation of Taylor Series method
     ```

6. **Milne's Predictor-Corrector Method**  
   - Uses past values to predict and correct the solution.
   - Predictor Formula:
     $y_{n+1} = y_{n-3} + \frac{4h}{3} (2 f_{n-1} - f_{n-2} + 2 f_n)$
   - Corrector Formula:
     $y_{n+1} = y_{n-1} + \frac{h}{3} (f_{n-1} + 4 f_n + f_{n+1})$
   - **Usage in Python:**
     ```python
     def milne_pc(f):
         ...  # Implementation of Milne's Predictor-Corrector method
     ```

7. **Adams-Bashforth Predictor-Corrector Method**  
   - A multi-step method using past function evaluations.
   - **Usage in Python:**
     ```python
     def adams_bashforth_pc(f):
         ...  # Implementation of Adams-Bashforth method
     ```

---

### 2nd Order ODE Method

#### Runge-Kutta 4th Order Method for 2nd Order ODEs
- Solves second-order ODEs of the form:
  $\frac{dy}{dx} = f(x, y, z), \quad \frac{dz}{dx} = g(x, y, z)$
- **Usage in Python:**
  ```python
  def rk4_second_order(f, g):
      ...  # Implementation of RK4 for second-order ODEs
  ```

---

## How to Use This Package
### Step 1: Clone the Repository
```bash
$ git clone https://github.com/yourusername/numerical-ode-methods.git
$ cd numerical-ode-methods
```

### Step 2: Install Dependencies
Make sure you have **SymPy** and **NumPy** installed:
```bash
$ pip install numpy sympy
```

### Step 3: Run Examples
You can test the methods using example differential equations. Example:
```python
import numpy as np
import sympy as smp
from (folder name where you have saved all 3 file) import euler, rk4 ... etc

def f(x, y):
    return x + y

rk4(f)
```

---

## Creating a Python Package
To create your own Python package:
1. Create a directory and add your `.py` files.
2. Create a `setup.py` file with metadata:
   ```python
   from setuptools import setup, find_packages
   setup(
       name='numerical_ode_methods',
       version='1.0',
       packages=find_packages(),
       install_requires=['numpy', 'sympy'],
   )
   ```
3. Install the package locally:
   ```bash
   $ pip install -e .
   ```
4. (Optional) Publish to PyPI:
   ```bash
   $ python setup.py sdist bdist_wheel
   $ twine upload dist/*
   ```

---

## Contributing
Feel free to contribute by submitting issues or pull requests.

---

## License
This project is done underb the guidence of Dr. RP singh and Dr. Harshita Madduri at Dept. of Mathematics NIT Kurukshetra.

