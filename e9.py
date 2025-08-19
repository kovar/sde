import numpy as np
import matplotlib.pyplot as plt

def run_bod_ensemble(K1, s1, sigma, B0, num_samples=100):
    """
    Runs an ensemble of BOD model simulations and plots the mean and standard deviation.

    Args:
        K1 (float): Decay rate.
        s1 (float): Source term.
        sigma (float): Volatility / magnitude of random noise.
        B0 (float): Initial BOD concentration.
        num_samples (int): The number of tracks to simulate.
    """
    # --- 1. Simulation Parameters ---
    T = 100.0
    num_steps = 100
    dt = T / num_steps

    # --- 2. Set up Arrays for Vectorized Simulation ---
    t = np.linspace(0, T, num_steps + 1)
    # Create a 2D array to hold all paths: (num_samples x num_steps)
    B = np.zeros((num_samples, num_steps + 1))
    B[:, 0] = B0  # Set initial condition for all samples

    # Generate all random increments at once
    dW = np.random.normal(0.0, np.sqrt(dt), (num_samples, num_steps))

    # --- 3. Run the Vectorized Euler-Maruyama Simulation ---
    for i in range(num_steps):
        # Calculate drift and diffusion for all samples at the current time step
        drift = (-K1 * B[:, i] + s1) * dt
        diffusion = sigma * dW[:, i]
        # Update all samples for the next time step
        B[:, i+1] = B[:, i] + drift + diffusion

    # --- 4. Calculate and Plot Statistics ---
    # Calculate mean across samples (axis=0) for each time point
    mean_B = np.mean(B, axis=0)
    # Calculate standard deviation across samples (axis=0)
    std_B = np.std(B, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(t, mean_B, label='Mean')
    plt.plot(t, std_B, label='Std. Dev.')
    plt.title(f"Mean and Standard Dev. of {num_samples} BOD Realizations\n(K1={K1}, s1={s1}, sigma={sigma})")
    plt.xlabel("Distance")
    plt.ylabel("Mean and Std. Dev.")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0, top=21) # Set axes for better comparison
    plt.show()

# --- Run Experiments with Different Parameter Values ---

# 1. Baseline Case: Replicates the figure in the exercise
print("Running baseline case (K1=0.2, s1=2.4, sigma=1.5)...")
run_bod_ensemble(K1=0.2, s1=2.4, sigma=1.5, B0=20)

# 2. Higher Volatility Case: Double the sigma
print("\nRunning case with higher volatility (sigma=3.0)...")
run_bod_ensemble(K1=0.2, s1=2.4, sigma=3.0, B0=20)

# 3. Faster Decay Rate Case: Increase K1
print("\nRunning case with faster decay/reversion (K1=0.8)...")
run_bod_ensemble(K1=0.8, s1=2.4, sigma=1.5, B0=20)

