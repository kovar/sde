import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot(b_val):
    """
    Generates and plots a realization of a Wiener process W(t) and
    a derived process X(t) = exp(b * W(t)).

    Args:
        b_val (float): The parameter 'b' for the process X(t).
    """
    # --- 1. Define Simulation Parameters ---
    T = 100.0          # Total time
    num_steps = 100    # Number of time steps (to match the exercise plot)
    dt = T / num_steps # Time step size

    # --- 2. Generate the Wiener Process W(t) ---
    t = np.linspace(0.0, T, num_steps + 1)
    # Generate the random increments dW ~ N(0, dt)
    dW = np.random.normal(0.0, np.sqrt(dt), num_steps)
    # Initialize the Wiener process array W with W(0) = 0
    W = np.zeros(num_steps + 1)
    # Cumulatively sum the increments to get the process path
    W[1:] = np.cumsum(dW)

    # --- 3. Generate the Derived Process X(t) ---
    # X(t) is defined as exp(b * W(t))
    X = np.exp(b_val * W)

    # --- 4. Plot the Results ---
    plt.figure(figsize=(10, 6))

    # Plot W(t) with a solid blue line
    plt.plot(t, W, 'b-', label='W(t)')

    # Plot X(t) with green dots, like in the exercise figure
    plt.plot(t, X, 'g.', markersize=4, label='X(t)')

    plt.title(f"Generation of a Stochastic Process for b = {b_val}")
    plt.xlabel("time t")
    plt.ylabel("W(t), X(t)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Generate realizations for various values of b ---
# For a smaller value of b
generate_and_plot(b_val=0.1)

# To replicate the figure in the exercise
generate_and_plot(b_val=0.3)

# For a larger value of b
generate_and_plot(b_val=0.8)

