import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot(a_val, b_val):
    """
    Generates and plots a realization of a generalized Wiener process:
    Xt = a*t + b*Wt

    Args:
        a_val (float): The drift coefficient.
        b_val (float): The volatility/diffusion coefficient.
    """
    # --- 1. Define Simulation Parameters ---
    T = 100.0          # Total time
    num_steps = 200    # Number of time steps for a smoother plot
    dt = T / num_steps # Time step size
    X0 = 0             # Initial condition

    # --- 2. Generate the Wiener Process W(t) ---
    t = np.linspace(0.0, T, num_steps + 1)
    # Generate the random increments dW ~ N(0, dt)
    dW = np.random.normal(0.0, np.sqrt(dt), num_steps)
    # Create the Wiener process path
    W = np.zeros(num_steps + 1)
    W[1:] = np.cumsum(dW)

    # --- 3. Generate the Generalized Wiener Process X(t) ---
    # Use the exact solution: Xt = X0 + a*t + b*W
    X = X0 + a_val * t + b_val * W

    # --- 4. Plot the Result ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, X)
    plt.title(f"Simulation of a Generalized Wiener Process (a={a_val}, b={b_val})")
    plt.xlabel("time t")
    plt.ylabel("X(t)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Run experiments with different parameter values ---

# 1. Replicate the figure in the exercise
print("Generating plot for a=0.2, b=0.3 (as in the exercise)...")
generate_and_plot(a_val=0.2, b_val=0.3)

# 2. No drift (a=0), same volatility
print("Generating plot for a=0, b=0.3 (no drift)...")
generate_and_plot(a_val=0, b_val=0.3)

# 3. Higher drift, same volatility
print("Generating plot for a=0.5, b=0.3 (higher drift)...")
generate_and_plot(a_val=0.5, b_val=0.3)

# 4. Same drift, higher volatility
print("Generating plot for a=0.2, b=1.0 (higher volatility)...")
generate_and_plot(a_val=0.2, b_val=1.0)

# 5. No stochasticity (a=0.2, b=0)
print("Generating plot for a=0.2, b=0 (no stochasticity)...")
generate_and_plot(a_val=0.2, b_val=0)
