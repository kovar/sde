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
    alpha_slider,
    mo,
    mu_slider,
    p_slider,
    scheme_dropdown,
    sigma0_slider,
    tracks_slider,
    xi0_slider,
):
    mo.md(
        rf"""
    ## Part a)

    _Implement the Euler and the Milstein strong order 1.0 scheme and generate numerical tracks for $0 \leq t \leq 1$ year for different values of $p$ and $\alpha$._

    Both schemes are implemented in the `run_simulation` function below. For the Black-Scholes model, volatility $\sigma$ is constant, so both parameters $p$ and $\alpha$ do not change the plotted tracks. The number of tracks to generate or the scheme can be adjusted below.

    | Parameter | Slider | Value |
    | --- | --- | --- |
    | $p$ | {p_slider} | {p_slider.value} |
    | $\alpha$ | {alpha_slider} | {alpha_slider.value} |
    | $\sigma_0$ | {sigma0_slider} | {sigma0_slider.value} |
    | $\xi_0$ | {xi0_slider} | {xi0_slider.value} |
    | $\mu$ | {mu_slider} | {mu_slider.value} |
    | Number of tracks to generate | {tracks_slider} | {tracks_slider.value} |
    | Scheme | {scheme_dropdown} | {scheme_dropdown.value} |
    """
    )
    return


@app.cell(hide_code=True)
def _(S, plt, scheme_dropdown, show_plot_svg, t):
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 6))

    for _i in range(S.shape[0]):
        _ax.plot(t, S[_i, :], lw=1.5, label=f"Track {_i + 1}")

    _ax.set_title(
        f"Simulated Stock Price Paths ({S.shape[0]} Tracks) - {scheme_dropdown.value} Scheme - Full Model"
    )
    _ax.set_xlabel("Time (Years) ($t$)")
    _ax.set_ylabel("Stock Price ($S_t$)")
    _ax.set_ylim(0, 100)
    _ax.grid(True)
    _ax.legend(loc="lower left")

    show_plot_svg(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Numerical Simulations for Full Model""")
    return


@app.cell
def _(
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
            "v": 0.5,  # Volatility of volatility
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "N": 500,  # Number of time steps
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
        mu, p, alpha, v = params["mu"], params["p"], params["alpha"], params["v"]
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
                    sigma_t - p * (sigma_t - xi_t) * dt + v * sigma_t * dW2[:, n]
                )
            elif scheme == "Milstein":
                # S update with Milstein correction
                S_correction = 0.5 * (sigma_t**2) * S_t * (dW1[:, n] ** 2 - dt)
                S[:, n + 1] = (
                    S_t + mu * S_t * dt + sigma_t * S_t * dW1[:, n] + S_correction
                )

                # Sigma update with Milstein correction
                sigma_correction = 0.5 * (v**2) * sigma_t * (dW2[:, n] ** 2 - dt)
                sigma[:, n + 1] = (
                    sigma_t
                    - p * (sigma_t - xi_t) * dt
                    + v * sigma_t * dW2[:, n]
                    + sigma_correction
                )
            else:
                raise ValueError("Scheme must be 'Euler' or 'Milstein'")

            # Xi is deterministic, so its update is always an Euler step
            xi[:, n + 1] = xi_t - alpha * (sigma_t - xi_t) * dt

            # Enforce positivity for S and sigma to prevent numerical instability
            S[:, n + 1] = np.maximum(S[:, n + 1], 0)
            sigma[:, n + 1] = np.maximum(sigma[:, n + 1], 0)

        return t, S, sigma, xi
    return (run_full_simulation,)


@app.cell
def _(get_full_model_params, run_full_simulation, scheme_dropdown):
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
def _(tracks_slider):
    def get_params():
        """Returns a dictionary of model and simulation parameters."""
        return {
            # Model Parameters
            "S0": 50.0,  # Initial stock price
            "mu": 0.10,  # Drift (annual)
            "sigma": 0.20,  # Volatility (annual)
            # Simulation Parameters
            "T": 1.0,  # Time horizon (1 year)
            "N": 252,  # Number of time steps (e.g., trading days in a year)
            "num_tracks": tracks_slider.value,  # Number of tracks to simulate
        }
    return (get_params,)


@app.cell
def _(get_params, run_simulation, scheme_dropdown):
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

    By varying the parameters with the sliders above, the following can be observed.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Helpers""")
    return


@app.cell
def _(mo):
    # sliders
    p_slider = mo.ui.slider(-100, 100, 0.1, value=50)
    alpha_slider = mo.ui.slider(-100, 100, 0.1, value=0.5)
    tracks_slider = mo.ui.slider(1, 10, 1, value=5)
    sigma0_slider = mo.ui.slider(0.01, 1.0, 0.01, value=0.20)
    xi0_slider = mo.ui.slider(0.01, 1.0, 0.01, value=0.20)
    mu_slider = mo.ui.slider(-1.0, 1.0, 0.01, value=0.10)

    # dropdowns
    scheme_dropdown = mo.ui.dropdown(
        ["Euler", "Milstein"],
        value="Euler",
    )
    return (
        alpha_slider,
        mu_slider,
        p_slider,
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
