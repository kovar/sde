import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Model and Simulation Parameters ---

# Model parameters
b = 0.2  # Constant from the SDE
X0 = 1.0  # Initial condition (since X_t = exp(b*W_t), X0 = exp(0) = 1)

# Simulation parameters
T = 10.0  # Final time, as mentioned in the plot's y-axis label
num_samples = 1000  # Number of samples to average for each dt, as per the text
num_experiments = 1  # How many times to repeat the whole experiment

# A list of time steps (dt) to test. Chosen to be similar to the figure.
dt_values = np.linspace(0.01, 0.2, 4)

# --- 2. Run the Full Experiment Multiple Times ---

plt.figure(figsize=(10, 7))

# Outer loop: Repeat the experiment to see statistical fluctuations
for experiment_num in range(num_experiments):
    print(f"Running experiment #{experiment_num + 1}...")
    mean_absolute_errors = []

    # Inner loop: Iterate over different time step sizes
    for dt in dt_values:
        num_steps = int(T / dt)
        errors_for_this_dt = []

        # Monte Carlo simulation loop
        for _ in range(num_samples):
            # Generate one Wiener path for this sample
            dW = np.random.normal(0.0, np.sqrt(dt), num_steps)
            W = np.sum(dW)  # We only need the final value W(T)

            # --- Calculate the Exact Solution at T ---
            # X(T) = exp(b * W(T))
            X_exact = np.exp(b * W)

            # --- Calculate the Numerical Solution at T using Euler-Maruyama ---
            X_numerical = X0
            for i in range(num_steps):
                drift = (b**2 / 2) * X_numerical * dt
                diffusion = b * X_numerical * dW[i]
                X_numerical += drift + diffusion

            # Store the absolute error for this sample
            errors_for_this_dt.append(np.abs(X_numerical - X_exact))

        # Average the errors over all samples and store the result
        mean_absolute_errors.append(np.mean(errors_for_this_dt))

    # Plot the error curve for this single experiment
    plt.plot(
        dt_values,
        mean_absolute_errors,
        marker="o",
        linestyle="-",
        label=f"Experiment {experiment_num + 1}",
    )

# --- 3. Finalize and Show Plot ---
plt.title("Error versus Timestep for Euler Scheme (Multiple Runs)")
plt.xlabel("Timestep Delta t")
plt.ylabel(f"Absolute Error at T={T}")
plt.legend()
plt.grid(True)
plt.show()
