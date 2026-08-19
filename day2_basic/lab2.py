import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
sp.init_printing(use_unicode=True)
 
x = sp.symbols('x', real=True)
# Choose a function (edit this line freely)
f_expr  = x**3 - 3*x                 # try sp.exp(-x**2) or sp.sin(x)*sp.exp(-0.2*x)
df_expr = sp.diff(f_expr, x)
 
# Turn symbolic expressions into fast numeric callables
f  = sp.lambdify(x, f_expr, "numpy")
df = sp.lambdify(x, df_expr, "numpy")
# Why: sympy.diff gets the exact derivative; lambdify produces NumPy‑friendly functions for plotting.

# 1) Plot the function f(x)f(x)f(x)
xs = np.linspace(-3, 3, 400)
ys = f(xs)
 
plt.figure()
plt.plot(xs, ys)
plt.title("Function f(x)")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.grid(True)
plt.show()
# 2) Plot the derivative f′(x)f'(x)f′(x)
ys_prime = df(xs)
 
plt.figure()
plt.plot(xs, ys_prime)
plt.axhline(0, linestyle='--')
plt.title("Derivative f'(x)")
plt.xlabel("x"); plt.ylabel("f'(x)"); plt.grid(True)
plt.show()
# Note: Where the derivative crosses zero, f(x)f(x)f(x) has horizontal tangents (candidate minima/maxima).

# 3) Tangent line at a point
x0 = -1.0  # try different values
y0 = f(x0); m0 = df(x0)
xs_zoom = np.linspace(x0-1.5, x0+1.5, 200)
tangent = y0 + m0 * (xs_zoom - x0)
 
plt.figure()
plt.plot(xs, ys, alpha=0.6)
plt.plot(xs_zoom, tangent)
plt.scatter([x0], [y0])
plt.title(f"Tangent at x0={x0} (slope={m0:.2f})")
plt.xlabel("x"); plt.ylabel("y"); plt.grid(True)
plt.show()
# 4) (Optional) Finite‑difference vs analytic derivative
h = 1e-3
num_deriv = (f(xs + h) - f(xs - h)) / (2*h)
 
plt.figure()
plt.plot(xs, ys_prime, label="analytic f'(x)")
plt.plot(xs, num_deriv, linestyle='--', label="finite-diff approx")
plt.legend(); plt.title("Analytic vs Numeric Derivative")
plt.xlabel("x"); plt.ylabel("derivative"); plt.grid(True)
plt.show()