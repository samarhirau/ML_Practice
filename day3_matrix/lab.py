# 1. Implement Gradient Descent for $f(x) = x^2$

import numpy as np

def gradient_descent(f_prime, x_init, lr=0.1, steps=20):
    x_history = [x_init]
    x = x_init
    
    for _ in range(steps):
        grad = f_prime(x)
        x = x - lr * grad
        x_history.append(x)
        
    return np.array(x_history)

# Derivative of f(x) = x^2
f_prime = lambda x: 2 * x

# Test run starting at x = 5.0
history = gradient_descent(f_prime, x_init=5.0, lr=0.1, steps=20)
print(f"Final value of x: {history[-1]:.6f}")

import matplotlib.pyplot as plt

x_init = 5.0
learning_rates = [0.01, 0.1, 1.0]

for lr in learning_rates:
    hist = gradient_descent(f_prime, x_init=x_init, lr=lr, steps=10)
    print(f"LR = {lr:<4} | First 5 steps: {np.round(hist[:5], 3)}")