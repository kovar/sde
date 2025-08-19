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
    mo.md(
        r"""
    ## Part b)

    _Study the influence of parameters and $p$ and $\alpha$ on the behaviour of $S_t$._

    See the report.
    """
    )
    return


@app.cell
def _(plt, show_plot_svg, stable_points, unstable_points):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    _ax.set_title(r"Numerical Stability Map for $p$ and $\alpha$")
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
    _ax.set_xlabel(r"$p$ (Speed of Volatility Reversion)")
    _ax.set_ylabel(r"$\alpha$ (Speed of Memory Adaptation)")
    _ax.set_xscale("log")  # Log scale is often useful for p
    _ax.set_yscale("log")  # and for alpha
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
        alpha_range=np.logspace(0, 2, 21),
        N=N_slider.value,
    )
    return stable_points, unstable_points


@app.cell
def _(mo):
    mo.md(r"""## Helpers""")
    return


@app.cell
def _(mo):
    # sliders
    p_slider = mo.ui.slider(0, 100, 0.1, value=50)
    alpha_slider = mo.ui.slider(0.1, 100, 0.1, value=0.5)
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
