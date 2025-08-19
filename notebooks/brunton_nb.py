import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo, n_slider, p_slider):
    mo.md(
        rf"""
    # Sliders

    | Parameter | Slider | Value |
    | --- | --- | --- |
    | n | {n_slider} | {n_slider.value} |
    | p | {p_slider} | {p_slider.value} |
    """
    )
    return


@app.cell
def _(n_slider, np, p_slider, plt, show_plot_svg, spsp):
    n = n_slider.value
    p = p_slider.value
    q = 1-p

    # Create Binomial distribution (exact)
    PXbinomial=[]
    for k in range(n+1):
        probX = spsp.comb(n,k) * p**k * q**(n-k)
        PXbinomial.append (probX)
    
    # Create Normal distribution (approximation)
    mu = n*p
    sigma = np.sqrt(n*p*q)
    X = np.arange(0, n, 0.1)
    PXnormal = (1/(sigma*np.sqrt(2 * np.pi))) * np.exp(-(X-mu)**2 / (2 * sigma ** 2))

    # Plot
    _fig, _ax1 = plt.subplots(1, 1, figsize=(7, 5))

    _ax1.plot(PXbinomial, label="Binomial Distribution", color='blue')
    _ax1.plot(X, PXnormal, label="Normal Approximation", color='orange')
    _ax1.legend()
    _ax1.set_xlim(0, n)
    _ax1.grid()

    show_plot_svg(_fig)
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(2, 100, 1, value=30)
    p_slider = mo.ui.slider(0.01, 0.99, 0.01, value=0.5)
    return n_slider, p_slider


@app.cell
def _(mo):
    mo.md(r"""# Helpers""")
    return


@app.cell
def _(io, mo, plt):
    def change_plot_style():
        """ this breaks the web version
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
    import scipy as sp
    import scipy.special as spsp
    import matplotlib.pyplot as plt
    return io, mo, np, plt, spsp


if __name__ == "__main__":
    app.run()
