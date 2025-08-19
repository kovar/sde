import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Model and Simulation Parameters ---
b = 0.5            # Constant parameter (volatility)
Phi0 = 1.0         # Initial condition

# Simulation parameters
T = 100.0          # Total time
num_steps = 1000   # Number of time steps for a smoother simulation
dt = T / num_steps   # Time step size

# --- 2. Generate a Single Wiener Process Path ---
# This path will be the common source of randomness for all solutions
t = np.linspace(0, T, num_steps + 1)
# Generate random increments
dW = np.random.normal(0.0, np.sqrt(dt), num_steps)
# Construct the Wiener path W(t)
W = np.zeros(num_steps + 1)
W[1:] = np.cumsum(dW)

# --- 3. Calculate the Three Different Solutions ---

# (a) The Exact Analytical Solution
# Phi_t = exp(b * W_t)
Phi_exact = np.exp(b * W)

# (b) The Correct Numerical Solution (Euler-Maruyama on the Itô SDE)
# SDE: dPhi = (b^2/2)*Phi*dt + b*Phi*dW
Phi_numerical = np.zeros(num_steps + 1)
Phi_numerical[0] = Phi0
for i in range(num_steps):
    ito_drift = (b**2 / 2) * Phi_numerical[i] * dt
    diffusion = b * Phi_numerical[i] * dW[i]
    Phi_numerical[i+1] = Phi_numerical[i] + ito_drift + diffusion

# (c) The "Naive" Numerical Solution (Incorrectly omitting the Itô term)
# Incorrect SDE: dPhi = b*Phi*dW
Phi_naive = np.zeros(num_steps + 1)
Phi_naive[0] = Phi0
for i in range(num_steps):
    naive_diffusion = b * Phi_naive[i] * dW[i]
    Phi_naive[i+1] = Phi_naive[i] + naive_diffusion

# --- 4. Plot the Results to Compare ---

# Plot 1: Comparing Exact and Correct Numerical Solution
plt.figure(figsize=(12, 7))
plt.plot(t, Phi_exact, 'b-', label='Exact Solution ($e^{bW_t}$)', linewidth=2.5)
plt.plot(t, Phi_numerical, 'g--', label='Numerical (with Itô term)', linewidth=1.5)
plt.title('Exact Solution vs. Correct Euler-Maruyama Numerical Solution')
plt.xlabel('time t')
plt.ylabel('$\\Phi(t)$')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Showing the Necessity of the Itô Term
plt.figure(figsize=(12, 7))
plt.plot(t, Phi_exact, 'b-', label='Exact Solution ($e^{bW_t}$)', linewidth=2.5)
plt.plot(t, Phi_naive, 'r--', label='Naive Numerical (NO Itô term)', linewidth=1.5)
plt.title('Experimental Proof: The Itô Term is Essential')
plt.xlabel('time t')
plt.ylabel('$\\Phi(t)$')
plt.legend()
plt.grid(True)
plt.show()
