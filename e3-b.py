import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
num_samples = 100  # Number of sample paths
T = 100.0          # Total time
num_steps = 100    # Number of time steps
dt = T / num_steps # Time step size

# --- Generate Multiple Wiener Process Paths ---
all_W = np.zeros((num_samples, num_steps + 1))
dW = np.random.normal(0.0, np.sqrt(dt), (num_samples, num_steps))
all_W[:, 1:] = np.cumsum(dW, axis=1)

# --- Calculate Mean and Variance ---
mean_W = np.mean(all_W, axis=0)
var_W = np.var(all_W, axis=0)

# --- Plot the Result ---
t = np.linspace(0.0, T, num_steps + 1)
plt.figure(figsize=(10, 6))

plt.plot(t, mean_W, 'b:', label='Mean')
plt.plot(t, var_W, 'g-', label='Variance')

plt.title(f"Mean and Variance of {num_samples} samples of standard Wiener Process")
plt.xlabel("time t")
plt.ylabel("Mean and Variance")
plt.legend()
plt.xlim(0, T)
plt.ylim(-20, 120)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
