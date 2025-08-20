import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Modelling the stock price assignment

    Prepared by Jiri Kovar, [@kovar](http://github.com/kovar)

    2025-08-19, Delft
    """
    )
    return


@app.cell(hide_code=True)
def _(
    N_slider,
    alpha_slider,
    mo,
    mu_slider,
    p_slider,
    regenerate_button,
    scheme_dropdown,
    sigma0_slider,
    tracks_slider,
    xi0_slider,
):
    mo.md(
        rf"""
    ## Part a)

    _Implement the Euler and the Milstein strong order 1.0 scheme and generate numerical tracks for $0 \leq t \leq 1$ year for different values of $p$ and $\alpha$._

    Both schemes are implemented in the cells below. For the Black-Scholes model, volatility $\sigma$ is constant, so both parameters $p$ and $\alpha$ do not change the plotted tracks. The initial conditions, number of tracks to generate, the scheme or the number of time steps can be adjusted below.

    | Parameter | Slider | Value | Parameter | Slider | Value |
    |---|---|---|---|---|---|
    | $p$ | {p_slider} | {p_slider.value} | Number of time steps | {N_slider} | {N_slider.value} |
    | $\alpha$ | {alpha_slider} | {alpha_slider.value} | Number of tracks to generate | {tracks_slider} | {tracks_slider.value} |
    | $\sigma_0$ | {sigma0_slider} | {sigma0_slider.value} | Scheme | {scheme_dropdown} | {scheme_dropdown.value} |
    | $\xi_0$ | {xi0_slider} | {xi0_slider.value} | Regenerate Plot | {regenerate_button} | Click to regenerate plot |
    | $\mu$ | {mu_slider} | {mu_slider.value} | | | |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Numerical Simulations for Full Model""")
    return


@app.cell(hide_code=True)
def _(S, plt, scheme_dropdown, show_plot_svg, t):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    for _i in range(S.shape[0]):
        _ax.plot(t, S[_i, :], lw=1.5, label=f"Track {_i + 1}")

    _ax.set_title(
        f"Simulated Stock Price Paths ({S.shape[0]} Tracks) - {scheme_dropdown.value} Scheme - Full Model"
    )
    _ax.set_xlabel("Time $t$ [years]")
    _ax.set_ylabel("Stock Price $S_t$ [€]")
    _ax.set_ylim(-1, 100)
    _ax.grid(True)
    _ax.legend(loc="lower left")

    show_plot_svg(_fig)
    return


@app.cell
def _(
    N_slider,
    alpha_slider,
    mu_slider,
    p_slider,
    sigma0_slider,
    tracks_slider,
    xi0_slider,
):
    def get_full_model_params():
        """Returns a dictionary of model and simulation parameters for the full model."""
        return {
            # Model Parameters
            "S0": 50.0,  # Initial stock price
            "sigma0": sigma0_slider.value,  # Initial volatility
            "xi0": xi0_slider.value,  # Initial long-term average volatility
            "mu": mu_slider.value,  # Drift (annual)
            "p": p_slider.value,  # Speed of reversion for sigma
            "alpha": alpha_slider.value,  # Speed of reversion for xi
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "N": N_slider.value,  # Number of time steps
            "num_tracks": tracks_slider.value,  # Number of paths to simulate
        }
    return (get_full_model_params,)


@app.cell
def _(np):
    def run_full_simulation(params, scheme="Euler"):
        """
        Runs a Monte Carlo simulation for the full 3-factor stochastic volatility model.
        """
        # Unpack parameters
        S0, sigma0, xi0 = params["S0"], params["sigma0"], params["xi0"]
        mu, p, alpha = params["mu"], params["p"], params["alpha"]
        T, N, num_tracks = params["T"], params["N"], params["num_tracks"]
        dt = T / N

        # Create arrays to store results
        t = np.linspace(0, T, N + 1)
        S = np.zeros((num_tracks, N + 1))
        sigma = np.zeros((num_tracks, N + 1))
        xi = np.zeros((num_tracks, N + 1))

        # Set initial conditions
        S[:, 0] = S0
        sigma[:, 0] = sigma0
        xi[:, 0] = xi0

        # Generate two independent sets of Wiener process increments
        dW1 = np.random.standard_normal((num_tracks, N)) * np.sqrt(dt)
        dW2 = np.random.standard_normal((num_tracks, N)) * np.sqrt(dt)

        # Run the simulation loop
        for n in range(N):
            # Get current values
            S_t, sigma_t, xi_t = S[:, n], sigma[:, n], xi[:, n]

            if scheme == "Euler":
                S[:, n + 1] = S_t + mu * S_t * dt + sigma_t * S_t * dW1[:, n]
                sigma[:, n + 1] = (
                    sigma_t - (sigma_t - xi_t) * dt + p * sigma_t * dW2[:, n]
                )
            elif scheme == "Milstein":
                # S update with Milstein correction
                S_correction = 0.5 * (sigma_t**2) * S_t * (dW1[:, n] ** 2 - dt)
                S[:, n + 1] = (
                    S_t + mu * S_t * dt + sigma_t * S_t * dW1[:, n] + S_correction
                )

                # sigma update with Milstein correction
                sigma_correction = 0.5 * p**2 * sigma_t * (dW2[:, n] ** 2 - dt)
                sigma[:, n + 1] = (
                    sigma_t
                    - (sigma_t - xi_t) * dt
                    + p * sigma_t * dW2[:, n]
                    + sigma_correction
                )
            else:
                raise ValueError("Scheme must be 'Euler' or 'Milstein'")

            # Xi is deterministic, so its update is always an Forward-Euler step
            xi[:, n + 1] = xi_t + 1 / alpha * (sigma_t - xi_t) * dt

            # Optional: enforce positivity for S and sigma to prevent numerical instability
            # S[:, n + 1] = np.maximum(S[:, n + 1], 0)  # Ensure stock price is non-negative
            # sigma[:, n + 1] = np.maximum(sigma[:, n + 1], 0)  # Ensure volatility is non-negative

        return t, S, sigma, xi
    return (run_full_simulation,)


@app.cell
def _(
    get_full_model_params,
    regenerate_button,
    run_full_simulation,
    scheme_dropdown,
):
    regenerate_button  # regenerates the plot when the regenerate button is pressed

    # Get parameters
    params = get_full_model_params()

    # Run simulation
    t, S, sigma, xi = run_full_simulation(params, scheme_dropdown.value)
    return S, t


@app.cell
def _(mo):
    mo.md(r"""### Numerical Simulations for Black-Scholes""")
    return


@app.cell
def _(N_slider, mu_slider, sigma0_slider, tracks_slider):
    def get_params():
        """Returns a dictionary of model and simulation parameters."""
        return {
            # Model Parameters
            "S0": 50.0,  # Initial stock price
            "mu": mu_slider.value,  # Drift (annual)
            "sigma": sigma0_slider.value,  # Volatility (annual)
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "N": N_slider.value,  # Number of time steps (e.g., trading days in a year)
            "num_tracks": tracks_slider.value,  # Number of tracks to simulate
        }
    return (get_params,)


@app.cell
def _(get_params, regenerate_button, run_simulation, scheme_dropdown):
    regenerate_button  # regenerates the plot when the regenerate button is pressed

    # Get parameters
    params_BS = get_params()

    # Run simulation
    t_BS, S_BS = run_simulation(params_BS, scheme_dropdown.value)
    return S_BS, t_BS


@app.cell
def _(np):
    def run_simulation(params, scheme="Euler"):
        """
        Runs a Monte Carlo simulation for the Black-Scholes model.

        Args:
            params (dict): A dictionary of model and simulation parameters.
            scheme (str): The numerical scheme to use ('euler' or 'milstein').

        Returns:
            A tuple (t, S) containing the time vector and the simulated price paths.
        """
        # Unpack parameters
        s0, mu, sigma = params["S0"], params["mu"], params["sigma"]
        T, N, num_tracks = params["T"], params["N"], params["num_tracks"]
        dt = T / N

        # Create arrays to store results
        t = np.linspace(0, T, N + 1)
        S = np.zeros((num_tracks, N + 1))
        S[:, 0] = s0

        # Generate random increments for the Wiener process
        # Shape: (num_tracks, N) for vectorized operations
        dW = np.random.standard_normal((num_tracks, N)) * np.sqrt(dt)

        # Run the simulation loop
        for n in range(N):
            # Current stock price (vector of all tracks)
            S_t = S[:, n]

            # Calculate drift and diffusion terms
            drift = mu * S_t * dt
            diffusion = sigma * S_t * dW[:, n]

            if scheme == "Euler":
                S[:, n + 1] = S_t + drift + diffusion
            elif scheme == "Milstein":
                correction = 0.5 * sigma**2 * S_t * (dW[:, n] ** 2 - dt)
                S[:, n + 1] = S_t + drift + diffusion + correction
            else:
                raise ValueError("Scheme must be 'Euler' or 'Milstein'")

        return t, S
    return (run_simulation,)


@app.cell
def _(S_BS, plt, scheme_dropdown, show_plot_svg, t_BS):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    for _i in range(S_BS.shape[0]):
        _ax.plot(t_BS, S_BS[_i, :], lw=1.5, label=f"Track {_i + 1}")

    _ax.set_title(
        f"Simulated Stock Price Paths ({S_BS.shape[0]} Tracks) - {scheme_dropdown.value} Scheme - Black-Scholes"
    )
    _ax.set_xlabel("Time (Years)")
    _ax.set_ylabel("Stock Price (S_t)")
    _ax.grid(True)
    _ax.legend()

    show_plot_svg(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""## Part b)""")
    return


@app.cell
def _(plt, scheme_dropdown, show_plot_svg, stable_points, unstable_points):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.set_title(
        rf"Numerical Stability Map for $p$ and $\alpha$ - {scheme_dropdown.value} Scheme"
    )
    _ax.scatter(
        stable_points["p"],
        stable_points["alpha"],
        color="green",
        label="Stable",
    )
    _ax.scatter(
        unstable_points["p"],
        unstable_points["alpha"],
        color="red",
        marker="x",
        label="Unstable",
    )
    _ax.set_xlabel(r"$p$")
    _ax.set_ylabel(r"$\alpha$")
    _ax.set_xscale("log")  # Log scale is often useful for p
    # _ax.set_yscale("log")  # and for alpha
    _ax.grid(True, ls="--", which="both")
    _ax.legend()

    show_plot_svg(_fig)
    return


@app.cell
def _(
    N_slider,
    get_full_model_params,
    np,
    run_full_simulation,
    scheme_dropdown,
):
    def analyze_stability(p_range, alpha_range, N=N_slider.value):
        """
        Analyzes the numerical stability of the full model over a grid of p and alpha values.

        Returns:
            A dictionary containing lists of stable and unstable (p, alpha) pairs.
        """
        stable_points = {"p": [], "alpha": []}
        unstable_points = {"p": [], "alpha": []}

        # Get base parameters and update number of tracks
        params = get_full_model_params()  # Do this for the Full Model only
        params["num_tracks"] = 1  # Only one track needed for assessment

        for p_val in p_range:
            for alpha_val in alpha_range:
                params["p"] = p_val
                params["alpha"] = alpha_val

                # Run a single simulation
                try:
                    t, S, sigma, xi = run_full_simulation(
                        params, scheme=scheme_dropdown.value
                    )

                    # Check for NaN or inf in the results
                    if np.isnan(S).any() or np.isinf(S).any():
                        unstable_points["p"].append(p_val)
                        unstable_points["alpha"].append(alpha_val)
                    else:
                        stable_points["p"].append(p_val)
                        stable_points["alpha"].append(alpha_val)
                except (ValueError, FloatingPointError):
                    # Catch errors that might arise during simulation
                    unstable_points["p"].append(p_val)
                    unstable_points["alpha"].append(alpha_val)

        return stable_points, unstable_points
    return (analyze_stability,)


@app.cell
def _(N_slider, analyze_stability, np):
    stable_points, unstable_points = analyze_stability(
        p_range=np.logspace(0, 2, 21),
        # alpha_range=np.logspace(0, 2, 21),
        alpha_range=np.linspace(1e-4, 1e-2, 21),
        N=N_slider.value,
    )
    return stable_points, unstable_points


@app.cell
def _(mo):
    mo.md(r"""## Part c)""")
    return


@app.cell
def _(
    mu_slider,
    np,
    scheme_dropdown,
    sigma0_slider,
    tracks_slider,
    xi0_slider,
):
    def get_BS_convergence_params():
        """Returns a dictionary of model and simulation parameters for the full model."""
        return {
            # Model Parameters
            "S0": 50.0,  # Initial stock price
            "sigma0": sigma0_slider.value,  # Initial volatility
            "xi0": xi0_slider.value,  # Initial long-term average volatility
            "mu": mu_slider.value,  # Drift (annual)
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "num_tracks": tracks_slider.value,  # Number of paths to simulate
            "N_list": np.logspace(
                4, 9, 6, base=2, dtype=int
            ),  # Step counts for convergence test
            "M": 10000,  # Number of Monte Carlo paths for averaging"
        }


    def simulate_BS_final_value(
        S0, mu, sigma, T, dW, scheme=scheme_dropdown.value
    ):
        """
        Simulates the final value S(T) for a single path of the Black-Scholes model using the given scheme.

        Parameters:
        - S0: Initial stock price
        - mu: Drift parameter
        - sigma: Constant volatility
        - T: Total time
        - dW: Array of Weiner process increments (must match the number of steps N, where dt = T/N)
        - scheme: 'Euler' or 'Milstein'

        Returns:
        - S_T: The final simulated stock price S(T)
        """
        N = len(dW)
        dt = T / N
        S = S0
        for n in range(N):
            drift = mu * S
            diffusion = sigma * S
            if scheme == "Euler":
                S += drift * dt + diffusion * dW[n]
            elif scheme == "Milstein":
                correction = 0.5 * diffusion * sigma * (dW[n] ** 2 - dt)
                S += drift * dt + diffusion * dW[n] + correction
            # Enforce positivity for real stock values ?
            # S = max(S, 0)
        return S


    def exact_BS_final_value(S0, mu, sigma, T, W_T):
        """
        Computes the exact final value S(T) for the Black-Scholes model given W_T.

        Parameters:
        - S0: Initial stock price
        - mu: Drift parameter
        - sigma: Constant volatility
        - T: Total time
        - W_T: The Wiener process value at T (sum of increments)

        Returns:
        - S_T_exact: The exact stock price at T
        """
        return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    return (
        exact_BS_final_value,
        get_BS_convergence_params,
        simulate_BS_final_value,
    )


@app.cell
def _(exact_BS_final_value, np, simulate_BS_final_value):
    def convergence_test_BS(params):
        """
        Runs a strong convergence test for the Black-Scholes model to verify the orders of the Euler (0.5) and Milstein (1.0) schemes.

        Parameters:
        - S0: Initial stock price
        - mu: Drift parameter
        - sigma: Constant volatility
        - T: Total time
        - N_list: List of step counts (e.g., [16, 32, 64, 128, 256, 512]), must be increasing powers of 2
        - M: Number of Monte Carlo paths for averaging

        Returns:
        - dt_array: Array of time steps corresponding to N_list
        - avg_errors_euler: Average strong errors for Euler scheme
        - avg_errors_milstein: Average strong errors for Milstein scheme
        """
        S0 = params["S0"]
        mu = params["mu"]
        sigma = params["sigma0"]
        T = params["T"]
        N_list = params["N_list"]
        M = params["M"]

        # Identify the finest resolution
        finest_N = max(N_list)
        dt_finest = T / finest_N

        # Initialize arrays to store average errors
        num_dt = len(N_list)
        avg_errors_euler = np.zeros(num_dt)
        avg_errors_milstein = np.zeros(num_dt)

        # Loop over M paths
        for m in range(M):
            # Generate Brownian increments at the finest resolution
            dW_finest = np.random.standard_normal(finest_N) * np.sqrt(dt_finest)

            # Compute W_T (same for all resolutions)
            W_T = np.sum(dW_finest)

            # Compute the exact S(T) (same for all resolutions)
            S_exact = exact_BS_final_value(S0, mu, sigma, T, W_T)

            # For each N in N_list, compute coarse increments and simulate
            for i, N in enumerate(N_list):
                dt = T / N
                step_size = finest_N // N  # Assumes N_list are powers of 2

                # Sum fine increments to get coarse increments
                dW_coarse = np.sum(dW_finest.reshape(N, step_size), axis=1)

                # Simulate numerical S(T) for Euler and Milstein
                S_euler = simulate_BS_final_value(
                    S0, mu, sigma, T, dW_coarse, scheme="Euler"
                )
                S_milstein = simulate_BS_final_value(
                    S0, mu, sigma, T, dW_coarse, scheme="Milstein"
                )

                # Compute absolute errors
                error_euler = np.abs(S_euler - S_exact)
                error_milstein = np.abs(S_milstein - S_exact)

                # Accumulate errors
                avg_errors_euler[i] += error_euler
                avg_errors_milstein[i] += error_milstein

        # Average over M paths
        avg_errors_euler /= M
        avg_errors_milstein /= M

        # Compute dt array
        dt_array = T / np.array(N_list)

        return dt_array, avg_errors_euler, avg_errors_milstein
    return (convergence_test_BS,)


@app.cell
def _(convergence_test_BS, get_BS_convergence_params):
    dt_array, avg_errors_euler, avg_errors_milstein = convergence_test_BS(
        get_BS_convergence_params()
    )
    return avg_errors_euler, avg_errors_milstein, dt_array


@app.cell
def _(avg_errors_euler, avg_errors_milstein, dt_array, np):
    # Calculate the convergence orders
    _order_euler = np.polyfit(np.log(dt_array), np.log(avg_errors_euler), 1)[0]
    _order_milstein = np.polyfit(np.log(dt_array), np.log(avg_errors_milstein), 1)[
        0
    ]

    print(f"Numerically calculated strong order for Euler: {_order_euler:.4f}")
    print(
        f"Numerically calculated strong order for Milstein: {_order_milstein:.4f}"
    )
    return


@app.cell
def _(avg_errors_euler, avg_errors_milstein, dt_array, plt, show_plot_svg):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.loglog(dt_array, avg_errors_euler, label="Euler (order 0.5)", marker="o")
    _ax.loglog(
        dt_array, avg_errors_milstein, label="Milstein (order 1.0)", marker="x"
    )
    _ax.set_title("Strong Convergence Test for Black-Scholes Model")
    _ax.set_xlabel(r"Time Step $\Delta t$")
    _ax.set_ylabel("Average Strong Error")
    _ax.legend()
    _ax.grid(True, which="both", ls="--")

    show_plot_svg(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""## Part d)""")
    return


@app.cell
def _(mu_slider, np, sigma0_slider, tracks_slider, xi0_slider):
    def get_BS_convergence_params_weak():
        """Returns a dictionary of model and simulation parameters for the full model."""
        return {
            # Model Parameters
            "S0": 50.0,  # Initial stock price
            "sigma0": sigma0_slider.value,  # Initial volatility
            "xi0": xi0_slider.value,  # Initial long-term average volatility
            "mu": mu_slider.value,  # Drift (annual)
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "num_tracks": tracks_slider.value,  # Number of paths to simulate
            "N_list": np.logspace(
                4, 9, 6, base=2, dtype=int
            ),  # Step counts for convergence test
            "M": 10000,  # Number of Monte Carlo paths for averaging"
        }


    def exact_BS_mean_final_value(S0, mu, T):
        """
        Computes the exact expected final value E[S(T)] for the Black-Scholes model.

        Parameters:
        - S0: Initial stock price
        - mu: Drift parameter
        - T: Total time

        Returns:
        - E_S_T: The exact expected stock price at T
        """
        return S0 * np.exp(mu * T)
    return (get_BS_convergence_params_weak,)


@app.cell
def _(exact_BS_final_value, np, simulate_BS_final_value):
    def weak_convergence_test_BS(params):
        """
        Runs a CORRECTED weak convergence test for the Black-Scholes model.
        This version uses common random numbers to dramatically reduce variance,
        revealing the true weak convergence trend.
        """
        # Unpack parameters
        S0 = params["S0"]
        mu = params["mu"]
        sigma = params["sigma0"]
        T = params["T"]
        N_list = params["N_list"]
        # With this better method, M doesn't need to be as massive.
        # 100,000 is more than enough.
        M = params["M"]

        # Initialize arrays to store the final weak errors
        num_dt = len(N_list)
        weak_errors_euler = np.zeros(num_dt)
        weak_errors_milstein = np.zeros(num_dt)

        # Loop over each time step resolution
        for i, N in enumerate(N_list):
            dt = T / N

            # Array to store the difference for each path
            path_errors_euler = np.zeros(M)
            path_errors_milstein = np.zeros(M)

            # Run M independent paths
            for m in range(M):
                # Generate a NEW random path for EACH simulation
                dW = np.random.standard_normal(N) * np.sqrt(dt)

                # For this path, calculate the exact final value
                W_T = np.sum(dW)
                S_exact = exact_BS_final_value(S0, mu, sigma, T, W_T)

                # Simulate the numerical final value using the SAME path
                S_euler = simulate_BS_final_value(S0, mu, sigma, T, dW, "Euler")
                S_milstein = simulate_BS_final_value(
                    S0, mu, sigma, T, dW, "Milstein"
                )

                # Store the error for this specific path
                path_errors_euler[m] = S_euler - S_exact
                path_errors_milstein[m] = S_milstein - S_exact

            # The weak error is the absolute value of the AVERAGE of the path errors
            weak_errors_euler[i] = np.abs(np.mean(path_errors_euler))
            weak_errors_milstein[i] = np.abs(np.mean(path_errors_milstein))

        # Compute dt array for plotting
        dt_array = T / np.array(N_list)

        return dt_array, weak_errors_euler, weak_errors_milstein
    return (weak_convergence_test_BS,)


@app.cell
def _(get_BS_convergence_params_weak, weak_convergence_test_BS):
    dt_weak, err_weak_euler, err_weak_milstein = weak_convergence_test_BS(
        get_BS_convergence_params_weak()
    )
    return dt_weak, err_weak_euler, err_weak_milstein


@app.cell
def _(
    dt_weak,
    err_weak_euler,
    err_weak_milstein,
    get_BS_convergence_params_weak,
    plt,
    show_plot_svg,
):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.loglog(dt_weak, err_weak_euler, label="Euler (order 1.0)", marker="o")
    _ax.loglog(
        dt_weak, err_weak_milstein, label="Milstein (order 1.0)", marker="x"
    )
    _ax.set_title(
        f"Weak Convergence Test for Black-Scholes Model - M = {get_BS_convergence_params_weak()['M']}"
    )
    _ax.set_xlabel(r"Time Step $\Delta t$")
    _ax.set_ylabel("Average Weak Error")
    _ax.legend()
    _ax.grid(True, which="both", ls="--")

    show_plot_svg(_fig)
    return


@app.cell
def _(dt_weak, err_weak_euler, err_weak_milstein, np):
    # Calculate the convergence orders
    _order_euler = np.polyfit(np.log(dt_weak), np.log(err_weak_euler), 1)[0]
    _order_milstein = np.polyfit(np.log(dt_weak), np.log(err_weak_milstein), 1)[0]

    print(f"Numerically calculated strong order for Euler: {_order_euler:.4f}")
    print(
        f"Numerically calculated strong order for Milstein: {_order_milstein:.4f}"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Part e)""")
    return


@app.cell
def _(np):
    def get_proxy_params():
        """
        Returns a dictionary of parameters specifically for calculating the
        high-accuracy 'proxy true' solutions.
        """
        # Base model parameters (can be linked to sliders), but here is fixed to avoid recalculation
        params = {
            "S0": 50.0,
            "sigma0": 0.20,
            "xi0": 0.20,
            "mu": 0.10,
            "p": 0.6,
            "alpha": 1.5,
            "T": 1.0,
            "N_list": np.logspace(
                4, 9, 6, base=2, dtype=int
            ),  # Step counts for convergence test
        }

        # Critical parameters for the proxy calculation
        # N_fine should be a large power of 2
        params["N_fine"] = 4096

        # M_fine is the number of paths for the weak proxy.
        # This needs to be large to get a stable mean. 1 million is a good start.
        params["M_fine"] = 10_000
        params["M_strong"] = 1_000  # Number of paths for strong convergence test
        params["M_weak"] = 100_000  # Number of paths for weak convergence test

        return params
    return (get_proxy_params,)


@app.cell
def _(np):
    def run_simulation_refactored(params, scheme="Euler", dW_arrays=None):
        """
        Refactored simulation engine for the Full Model.

        It can either generate its own random increments or use a pre-supplied
        set, which is essential for the strong convergence test.

        Args:
            params (dict): A dictionary of model and simulation parameters.
            scheme (str): The numerical scheme to use ('euler' or 'milstein').
            dW_arrays (tuple, optional): A tuple (dW1, dW2) of pre-generated
                                         Wiener increments. If None, they will be
                                         generated internally. Defaults to None.

        Returns:
            A tuple (t, S, sigma, xi) containing the simulation results.
        """
        # Unpack parameters
        S0, sigma0, xi0 = params["S0"], params["sigma0"], params["xi0"]
        mu, p, alpha = params["mu"], params["p"], params["alpha"]
        T, N, num_tracks = params["T"], params["N"], params["num_tracks"]
        dt = T / N

        # Random Number Generation
        if dW_arrays is None:
            # If no arrays are provided, generate them internally.
            # This is used for the weak convergence test.
            dW1 = np.random.standard_normal((num_tracks, N)) * np.sqrt(dt)
            dW2 = np.random.standard_normal((num_tracks, N)) * np.sqrt(dt)
        else:
            # If arrays are provided, unpack and use them.
            # This is used for the strong convergence test.
            dW1, dW2 = dW_arrays
            # Ensure the shape is compatible if num_tracks > 1
            if num_tracks > 1 and dW1.ndim == 1:
                dW1 = np.tile(dW1, (num_tracks, 1))
                dW2 = np.tile(dW2, (num_tracks, 1))

        # Array Initialization
        t = np.linspace(0, T, N + 1)
        S = np.zeros((num_tracks, N + 1))
        S[:, 0] = S0
        sigma = np.zeros((num_tracks, N + 1))
        sigma[:, 0] = sigma0
        xi = np.zeros((num_tracks, N + 1))
        xi[:, 0] = xi0

        # Simulation Loop
        for n in range(N):
            S_t, sigma_t, xi_t = S[:, n], sigma[:, n], xi[:, n]

            # Define drift and diffusion terms
            S_drift = mu * S_t
            S_diffusion = sigma_t * S_t
            sigma_drift = -(sigma_t - xi_t)
            sigma_diffusion = p * sigma_t
            xi_drift = 1 / alpha * (sigma_t - xi_t)

            if scheme == "Euler":
                S[:, n + 1] = S_t + S_drift * dt + S_diffusion * dW1[:, n]
                sigma[:, n + 1] = (
                    sigma_t + sigma_drift * dt + sigma_diffusion * dW2[:, n]
                )
            elif scheme == "Milstein":
                S_correction = 0.5 * S_t * sigma_t**2 * (dW1[:, n] ** 2 - dt)
                S[:, n + 1] = (
                    S_t + S_drift * dt + S_diffusion * dW1[:, n] + S_correction
                )

                sigma_correction = 0.5 * sigma_t * p**2 * (dW2[:, n] ** 2 - dt)
                sigma[:, n + 1] = (
                    sigma_t
                    + sigma_drift * dt
                    + sigma_diffusion * dW2[:, n]
                    + sigma_correction
                )

            xi[:, n + 1] = xi_t + xi_drift * dt

            S[:, n + 1] = np.maximum(S[:, n + 1], 0)
            sigma[:, n + 1] = np.maximum(sigma[:, n + 1], 0)

        return t, S, sigma, xi
    return (run_simulation_refactored,)


@app.cell
def _(np, run_simulation_refactored):
    def calculate_strong_proxy(params, verbose=False):
        """
        Calculates the 'proxy true' benchmark for a STRONG convergence test.

        This involves simulating a single path with a very fine time step using the
        Milstein scheme and returning both its final value and the Wiener increments
        that generated it.

        Args:
            params (dict): A dictionary with model parameters and N_fine.

        Returns:
            A tuple containing:
            - S_proxy_final (float): The final S(T) of the fine path.
            - dW_arrays_fine (tuple): The (dW1, dW2) arrays for the fine path.
        """
        # Set up parameters for a single, fine simulation
        sim_params = params.copy()
        sim_params["num_tracks"] = 1
        sim_params["N"] = params["N_fine"]

        # Generate the master path of random numbers
        dt_fine = params["T"] / params["N_fine"]
        dW1_fine = np.random.standard_normal((1, params["N_fine"])) * np.sqrt(
            dt_fine
        )
        dW2_fine = np.random.standard_normal((1, params["N_fine"])) * np.sqrt(
            dt_fine
        )

        # Simulate using the Milstein scheme and the pre-generated numbers
        _, S_fine, _, _ = run_simulation_refactored(
            sim_params, scheme="Milstein", dW_arrays=(dW1_fine, dW2_fine)
        )

        # The final value of the simulation is our proxy
        S_proxy_final = S_fine[0, -1]

        if verbose:
            print(f"Final S(T) for strong proxy: {S_proxy_final:.4f}")

        return S_proxy_final, (dW1_fine, dW2_fine)


    def calculate_weak_proxy(params, verbose=False):
        """
        Calculates the 'proxy true' benchmark for a WEAK convergence test.

        This involves running a very large Monte Carlo simulation with a fine time
        step using the Milstein scheme and returning the mean of all final values.

        Args:
            params (dict): A dictionary with model parameters, N_fine, and M_fine.

        Returns:
            E_proxy_weak (float): The calculated mean E[S(T)].
        """
        print(
            f"Calculating weak proxy with M_fine = {params['M_fine']} and N_fine = {params['N_fine']}..."
        )
        print("(This may take a significant amount of time)")

        # Set up parameters for the massive simulation
        sim_params = params.copy()
        sim_params["num_tracks"] = params["M_fine"]
        sim_params["N"] = params["N_fine"]

        # Run the simulation. The engine will generate its own random numbers
        # because we are not passing the dW_arrays argument.
        _, S_paths, _, _ = run_simulation_refactored(sim_params, scheme="Milstein")

        # The proxy is the average of the final values of all simulated paths
        E_proxy_weak = np.mean(S_paths[:, -1])

        if verbose:
            print(f"Calculated E[S(T)] for weak proxy: {E_proxy_weak:.4f}")

        return E_proxy_weak
    return calculate_strong_proxy, calculate_weak_proxy


@app.cell
def _(calculate_strong_proxy, calculate_weak_proxy, get_proxy_params):
    S_proxy_final, dW_arrays_fine = calculate_strong_proxy(get_proxy_params())
    E_proxy_weak = calculate_weak_proxy(get_proxy_params())
    return


@app.cell
def _(calculate_strong_proxy, np, run_simulation_refactored):
    def strong_convergence_test_full(params):
        """
        Runs a strong convergence test for the full stochastic volatility model.
        This version contains the fix for the array dimension IndexError.
        """
        N_list = params["N_list"]
        # Use the correct M value for the strong test
        M = params["M_strong"]
        N_fine = params["N_fine"]
        T = params["T"]

        num_dt = len(N_list)
        total_errors_euler = np.zeros((M, num_dt))
        total_errors_milstein = np.zeros((M, num_dt))

        print(
            f"Running strong convergence test for the full model with M={M} paths..."
        )

        # --- Main Monte Carlo Loop ---
        for m in range(M):
            if (m + 1) % (M // 10) == 0:
                print(f"  ...completed {m + 1}/{M} paths")

            # STEP 1: For each path, generate a new proxy solution
            S_proxy_final, dW_arrays_fine = calculate_strong_proxy(params)
            dW1_fine, dW2_fine = dW_arrays_fine

            # STEP 2: Loop through coarser time steps for comparison
            for i, N in enumerate(N_list):
                sim_params = params.copy()
                sim_params["num_tracks"] = 1
                sim_params["N"] = N

                # --- Create coarse increments from the fine ones ---
                step_size = N_fine // N
                dW1_coarse_1D = np.sum(dW1_fine.reshape(-1, step_size), axis=1)
                dW2_coarse_1D = np.sum(dW2_fine.reshape(-1, step_size), axis=1)

                # --- Reshape the 1D arrays to 2D arrays of shape (1, N) ---
                dW1_coarse = dW1_coarse_1D.reshape(1, -1)
                dW2_coarse = dW2_coarse_1D.reshape(1, -1)

                # --- Run Euler simulation with coarse increments ---
                _, S_euler, _, _ = run_simulation_refactored(
                    sim_params, scheme="Euler", dW_arrays=(dW1_coarse, dW2_coarse)
                )

                # --- Run Milstein simulation with coarse increments ---
                _, S_milstein, _, _ = run_simulation_refactored(
                    sim_params,
                    scheme="Milstein",
                    dW_arrays=(dW1_coarse, dW2_coarse),
                )

                # STEP 3: Calculate and store the error against the proxy
                total_errors_euler[m, i] = np.abs(S_euler[0, -1] - S_proxy_final)
                total_errors_milstein[m, i] = np.abs(
                    S_milstein[0, -1] - S_proxy_final
                )

        # STEP 4: Average the errors over all M paths
        avg_errors_euler = np.mean(total_errors_euler, axis=0)
        avg_errors_milstein = np.mean(total_errors_milstein, axis=0)

        dt_array = T / np.array(N_list)

        return dt_array, avg_errors_euler, avg_errors_milstein
    return


@app.cell
def _(calculate_strong_proxy, np, run_simulation_refactored):
    def weak_convergence_test_full_1(params, E_proxy_weak):
        """
        Runs a weak convergence test for the full stochastic volatility model.

        It compares the mean of many simulations against a pre-calculated,
        high-accuracy 'proxy true' mean.

        Args:
            params (dict): Dictionary with model and simulation parameters (N_list, M).
            E_proxy_weak (float): The pre-calculated proxy mean E[S(T)].

        Returns:
            A tuple (dt_array, weak_errors_euler, weak_errors_milstein).
        """
        N_list = params["N_list"]
        M = params["M_weak"]
        T = params["T"]

        num_dt = len(N_list)
        weak_errors_euler = np.zeros(num_dt)
        weak_errors_milstein = np.zeros(num_dt)

        print(
            f"\nRunning weak convergence test for the full model with M={M} paths..."
        )

        # Loop over each time step resolution
        for i, N in enumerate(N_list):
            print(f"  ...testing for N={N} (dt={T / N:.4f})")
            sim_params = params.copy()
            sim_params["num_tracks"] = M
            sim_params["N"] = N

            # --- Euler: Get the mean of M simulations ---
            _, S_euler_paths, _, _ = run_simulation_refactored(
                sim_params, scheme="Euler"
            )
            E_numerical_euler = np.mean(S_euler_paths[:, -1])
            weak_errors_euler[i] = np.abs(E_numerical_euler - E_proxy_weak)

            # --- Milstein: Get the mean of M simulations ---
            _, S_milstein_paths, _, _ = run_simulation_refactored(
                sim_params, scheme="Milstein"
            )
            E_numerical_milstein = np.mean(S_milstein_paths[:, -1])
            weak_errors_milstein[i] = np.abs(E_numerical_milstein - E_proxy_weak)

        dt_array = T / np.array(N_list)

        return dt_array, weak_errors_euler, weak_errors_milstein


    def weak_convergence_test_full_2(params):
        """
        Runs a weak convergence test for the full model.

        This version uses common random numbers and path-wise differences to
        reduce variance and reveal the true weak convergence trend.
        """
        N_list = params["N_list"]
        M = params["M_weak"]
        N_fine = params["N_fine"]
        T = params["T"]

        num_dt = len(N_list)
        # Store the path-wise errors for averaging later
        total_errors_euler = np.zeros((M, num_dt))
        total_errors_milstein = np.zeros((M, num_dt))

        print(
            f"\nRunning weak convergence test for the full model with M={M} paths..."
        )

        # --- Main Monte Carlo Loop ---
        for m in range(M):
            if (m + 1) % (M // 10) == 0:
                print(f"  ...completed {m + 1}/{M} paths")

            # STEP 1: For each path, generate a new proxy solution
            S_proxy_final, dW_arrays_fine = calculate_strong_proxy(
                params, verbose=False
            )
            dW1_fine, dW2_fine = dW_arrays_fine

            # STEP 2: Loop through coarser time steps for comparison
            for i, N in enumerate(N_list):
                sim_params = params.copy()
                sim_params["num_tracks"] = 1
                sim_params["N"] = N

                # --- Create coarse increments from the fine ones ---
                step_size = N_fine // N
                dW1_coarse_1D = np.sum(dW1_fine.reshape(-1, step_size), axis=1)
                dW2_coarse_1D = np.sum(dW2_fine.reshape(-1, step_size), axis=1)
                dW1_coarse = dW1_coarse_1D.reshape(1, -1)
                dW2_coarse = dW2_coarse_1D.reshape(1, -1)

                # --- Run simulations with coarse increments ---
                _, S_euler, _, _ = run_simulation_refactored(
                    sim_params, scheme="Euler", dW_arrays=(dW1_coarse, dW2_coarse)
                )
                _, S_milstein, _, _ = run_simulation_refactored(
                    sim_params,
                    scheme="Milstein",
                    dW_arrays=(dW1_coarse, dW2_coarse),
                )

                # STEP 3: Store the SIGNED difference for this path
                total_errors_euler[m, i] = S_euler[0, -1] - S_proxy_final
                total_errors_milstein[m, i] = S_milstein[0, -1] - S_proxy_final

        # STEP 4: Average the differences, THEN take the absolute value
        avg_errors_euler = np.abs(np.mean(total_errors_euler, axis=0))
        avg_errors_milstein = np.abs(np.mean(total_errors_milstein, axis=0))

        dt_array = T / np.array(N_list)

        return dt_array, avg_errors_euler, avg_errors_milstein
    return


@app.cell
def _():
    # dt_strong_full, err_strong_euler_full, err_strong_milstein_full = (
    #     strong_convergence_test_full(get_proxy_params())
    # )
    return


@app.cell
def _(dt_strong_full, err_strong_euler_full, err_strong_milstein_full, np):
    # Calculate the convergence orders
    _order_euler = np.polyfit(
        np.log(dt_strong_full), np.log(err_strong_euler_full), 1
    )[0]
    _order_milstein = np.polyfit(
        np.log(dt_strong_full), np.log(err_strong_milstein_full), 1
    )[0]

    print(f"Numerically calculated strong order for Euler: {_order_euler:.4f}")
    print(
        f"Numerically calculated strong order for Milstein: {_order_milstein:.4f}"
    )
    return


@app.cell
def _(
    dt_strong_full,
    err_strong_euler_full,
    err_strong_milstein_full,
    get_proxy_params,
    plt,
    show_plot_svg,
):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.loglog(
        dt_strong_full,
        err_strong_euler_full,
        label="Euler (order 0.5)",
        marker="o",
    )
    _ax.loglog(
        dt_strong_full,
        err_strong_milstein_full,
        label="Milstein (order 0.5)",
        marker="x",
    )
    _ax.set_title(
        f"Strong Convergence Test for Black-Scholes Model - M = {get_proxy_params()['M_strong']}, N = {get_proxy_params()['N_fine']}"
    )
    _ax.set_xlabel(r"Time Step $\Delta t$")
    _ax.set_ylabel("Average Strong Error")
    _ax.legend()
    _ax.grid(True, which="both", ls="--")

    show_plot_svg(_fig)
    return


@app.cell
def _():
    # dt_weak_full, err_weak_euler_full, err_weak_milstein_full = (
    #     weak_convergence_test_full_2(get_proxy_params())
    # )
    return


@app.cell
def _(dt_weak_full, err_weak_euler_full, err_weak_milstein_full, np):
    # Calculate the convergence orders
    _order_euler = np.polyfit(
        np.log(dt_weak_full), np.log(err_weak_euler_full), 1
    )[0]
    _order_milstein = np.polyfit(
        np.log(dt_weak_full), np.log(err_weak_milstein_full), 1
    )[0]

    print(f"Numerically calculated strong order for Euler: {_order_euler:.4f}")
    print(
        f"Numerically calculated strong order for Milstein: {_order_milstein:.4f}"
    )
    return


@app.cell
def _(
    dt_weak_full,
    err_weak_euler_full,
    err_weak_milstein_full,
    get_proxy_params,
    plt,
    show_plot_svg,
):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.loglog(
        dt_weak_full, err_weak_euler_full, label="Euler (order 1.0)", marker="o"
    )
    _ax.loglog(
        dt_weak_full,
        err_weak_milstein_full,
        label="Milstein (order 1.0)",
        marker="x",
    )
    _ax.set_title(
        f"Weak Convergence Test for Full Model - M = {get_proxy_params()['M_weak']}"
    )
    _ax.set_xlabel(r"Time Step $\Delta t$")
    _ax.set_ylabel("Average Weak Error")
    _ax.legend()
    _ax.grid(True, which="both", ls="--")

    show_plot_svg(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""## Helpers""")
    return


@app.cell
def _(mo):
    # sliders
    p_slider = mo.ui.slider(-20, 20, 0.1, value=1.0)
    alpha_slider = mo.ui.slider(0.001, 10, 0.1, value=0.5)
    tracks_slider = mo.ui.slider(1, 10, 1, value=5)
    sigma0_slider = mo.ui.slider(0.01, 1.0, 0.01, value=0.20)
    xi0_slider = mo.ui.slider(0.01, 1.0, 0.01, value=0.20)
    mu_slider = mo.ui.slider(-1.0, 1.0, 0.01, value=0.10)
    N_slider = mo.ui.slider(10, 1000, 1, value=252)

    # dropdowns
    scheme_dropdown = mo.ui.dropdown(
        ["Euler", "Milstein"],
        value="Euler",
    )

    # buttons
    regenerate_button = mo.ui.button()
    return (
        N_slider,
        alpha_slider,
        mu_slider,
        p_slider,
        regenerate_button,
        scheme_dropdown,
        sigma0_slider,
        tracks_slider,
        xi0_slider,
    )


@app.cell
def _(io, mo, plt):
    def change_plot_style():
        """this breaks the web version
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Computer Modern Roman"
        })
        """
        pass


    def convert_fig_to_svg(fig):
        # Save the plot to an in-memory SVG buffer
        svg_buffer = io.StringIO()
        fig.savefig(svg_buffer, format="svg")
        svg_buffer.seek(0)
        svg_data = svg_buffer.getvalue()
        return svg_data


    def plot_svg(svg_data):
        # Display the SVG in Marimo using mo.Html
        return mo.Html(f"""
            <div>
                {svg_data}
            </div>
        """)


    def show_plot_svg(fig):
        # Close the default plot to prevent default PNG rendering
        plt.close()

        # Change plotting style
        change_plot_style()

        # Display the SVG in Marimo using mo.Html
        _svg_data = convert_fig_to_svg(fig)
        return plot_svg(_svg_data)
    return (show_plot_svg,)


@app.cell
def _():
    import marimo as mo
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    return io, mo, np, plt


if __name__ == "__main__":
    app.run()
