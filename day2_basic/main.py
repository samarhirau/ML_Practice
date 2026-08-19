import sympy as sp

sp.init_printing(use_unicode=True)
 
x, y = sp.symbols('x y', real=True)

f1 = x**3 + 2*x**2 - 5*x + 7
sp.diff(f1, x)

f2 = sp.sin(x) * sp.exp(x)
sp.simplify(sp.diff(f2, x))

f3 = sp.cos(x)
sp.diff(f3, x)        # first
sp.diff(f3, x, 2)     # second
sp.diff(f3, x, 3)     # third

f_prod  = (x**2 + 1) * sp.sin(x)
df_prod = sp.simplify(sp.diff(f_prod, x))
 
f_chain  = sp.log(1 + x**2)
df_chain = sp.simplify(sp.diff(f_chain, x))

f_xy = x**2 * y + sp.sin(x*y)
sp.diff(f_xy, x)         # ∂f/∂x
sp.diff(f_xy, y)         # ∂f/∂y
sp.diff(f_xy, x, y)      # mixed partial ∂²f/∂y∂x

# 5) Evaluate derivatives at points
df1 = sp.diff(x**3 + 2*x**2 - 5*x + 7, x)
df1.subs({x: 2})
 
fx = sp.diff(f_xy, x)
fx.subs({x: 1, y: 2})
# Why: .subs plugs numbers into symbolic expressions.

import numpy as np, matplotlib.pyplot as plt
 
g  = sp.exp(-x**2)
dg = sp.diff(g, x)
 
g_num  = sp.lambdify(x, g, "numpy")
dg_num = sp.lambdify(x, dg, "numpy")
 
xs = np.linspace(-3, 3, 200)
plt.figure()
plt.plot(xs, g_num(xs), label='g(x)=e^{-x^2}')
plt.plot(xs, dg_num(xs), label="g'(x)")
plt.legend(); plt.title('Function and its derivative'); plt.grid(True)
plt.show()