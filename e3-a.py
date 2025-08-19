import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
T = 100.0          # Total time
num_steps = 100    # Number of time steps
dt = T / num_steps # Time step size

# --- Generate a Single Wiener Process ---
t = np.linspace(0.0, T, num_steps + 1)
dW = np.random.normal(0.0, np.sqrt(dt), num_steps)
W = np.zeros(num_steps + 1)
W[1:] = np.cumsum(dW)

# --- Plot the Result ---
plt.figure(figsize=(10, 6))
plt.plot(t, W)
plt.title("Simulation of a single sample of standard Wiener Process")
plt.xlabel("time t")
plt.ylabel("W(t)")
plt.xlim(0, T)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
