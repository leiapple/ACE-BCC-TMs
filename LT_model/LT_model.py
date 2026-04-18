import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.optimize import root_scalar
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt, root_scalar


@app.cell
def _(mo):
    mo.md("""
    ## First define the function for LT model, then use the kink pair formation energy and Peierls stress from MD to find R and alpha, and finally plot the predicted ΔH*(τ) curve.
    """)
    return


@app.cell
def _(np, plt, root_scalar):
    # -----------------------------
    # Helper functions
    # -----------------------------
    def gamma_func(x, Gamma_c, Gamma_0, alpha, a):
        return (
            (Gamma_c + Gamma_0) / 2
            + (Gamma_c - Gamma_0) / 2
            * (
                alpha / 4
                + np.cos(2 * np.pi * x / a)
                - alpha / 4 * np.cos(4 * np.pi * x / a)
            )
        )


    def find_first_root_in_interval(fun, x_left, x_right, nscan=10000, tol=1e-10):
        """
        Find the first root of fun in [x_left, x_right] by scanning for sign changes
        and then refining with brentq.
        """
        xs = np.linspace(x_left, x_right, nscan)
        vals = np.array([fun(x) for x in xs])

        for i in range(len(xs) - 1):
            f1 = vals[i]
            f2 = vals[i + 1]

            if np.isnan(f1) or np.isnan(f2):
                continue

            if abs(f1) < tol:
                return xs[i]

            if f1 * f2 < 0:
                sol = root_scalar(fun, bracket=[xs[i], xs[i + 1]], method="brentq")
                if sol.converged:
                    return sol.root

        raise RuntimeError(f"No root found in [{x_left}, {x_right}].")

    def find_nontrivial_root_in_interval(fun, x_left, x_right, nscan=20000):
        xs = np.linspace(x_left, x_right, nscan)
        vals = np.array([fun(x) for x in xs])

        for i in range(len(xs) - 1):
            f1 = vals[i]
            f2 = vals[i + 1]

            if np.isnan(f1) or np.isnan(f2):
                continue

            # only use sign change, not abs(f)<tol
            if f1 * f2 < 0:
                sol = root_scalar(fun, bracket=[xs[i], xs[i + 1]], method="brentq")
                if sol.converged:
                    return sol.root

        raise RuntimeError(f"No nontrivial root found in [{x_left}, {x_right}].")

    def LT_model(DEp_b, R, b, alpha):
        # -----------------------------
        # Constants
        # -----------------------------
        eVtoGPa = 160.21766208
        Nsteps = 10000

        a_b = 2 * b / np.sqrt(3)
        a = np.sqrt(2 / 3) * a_b
        DEp = DEp_b / b
        Gamma_0 = DEp / (R - 1)
        Gamma_c = Gamma_0 * R

        if abs(alpha) != 0:
            tau_P = (
                np.pi * (Gamma_c - Gamma_0)
                / (16 * abs(alpha) * a * b)
                * (3 + np.sqrt(1 + 8 * alpha**2))
                * np.sqrt(8 * alpha**2 - 2 + 2 * np.sqrt(1 + 8 * alpha**2))
            )
        else:
            tau_P = np.pi * (Gamma_c - Gamma_0) / (a * b)

        K = np.arange(0.001, 0.999, 0.01)

        # -----------------------------
        # Arrays for results
        # -----------------------------
        y0 = np.zeros(len(K))
        lambda_c = np.zeros(len(K))
        Dy = np.zeros(len(K))
        Gamma_y0 = np.zeros(len(K))
        Un = np.zeros(len(K))

        # -----------------------------
        # Main loop
        # -----------------------------
        for i, Ki in enumerate(K):

            # Step 1 - y0
            def fun_y(y):
                return (
                    Ki * tau_P * b
                    + np.pi / a
                    * (Gamma_c - Gamma_0)
                    * np.sin(2 * np.pi * y / a)
                    * (1 - alpha * np.cos(2 * np.pi * y / a))
                )

            # choose the left root in [-a/2, 0], matching the MATLAB behavior near -a/2
            y0[i] = find_first_root_in_interval(fun_y, -a / 2, 0.0)

            # Step 2 - lambda_c
            def fun_lambda(lam):
                return (
                    (Gamma_c - Gamma_0) / 2
                    * (np.cos(2 * np.pi * lam / a) - np.cos(2 * np.pi * y0[i] / a))
                    - alpha / 8
                    * (Gamma_c - Gamma_0)
                    * (np.cos(4 * np.pi * lam / a) - np.cos(4 * np.pi * y0[i] / a))
                    - Ki * tau_P * b * (lam - y0[i])
                )

            # avoid the trivial root lambda = y0
            #eps = 1e-4
            #lambda_c[i] = find_first_root_in_interval(fun_lambda, y0[i] + eps, a / 2)

            delta = max(1e-3 * a, 1e-4)
            lambda_c[i] = find_nontrivial_root_in_interval(fun_lambda, y0[i] + delta, a / 2)

            # Step 3 - kink-pair nucleation energy
            Dy[i] = (lambda_c[i] - y0[i]) / Nsteps
            Gamma_y0[i] = gamma_func(y0[i], Gamma_c, Gamma_0, alpha, a)

            Un_d = 0.0
            for j in range(Nsteps):
                y01 = y0[i] + j * Dy[i]
                y02 = y0[i] + (j + 1) * Dy[i]

                Gamma_y01 = gamma_func(y01, Gamma_c, Gamma_0, alpha, a)
                Gamma_y02 = gamma_func(y02, Gamma_c, Gamma_0, alpha, a)

                rad1 = Gamma_y01**2 - (Ki * tau_P * b * (y01 - y0[i]) + Gamma_y0[i]) ** 2
                rad2 = Gamma_y02**2 - (Ki * tau_P * b * (y02 - y0[i]) + Gamma_y0[i]) ** 2

                tol = 1e-8

                if rad1 < -tol:
                    print(f"Warning: rad1 significantly negative at i={i}, j={j}: {rad1}")
                if rad2 < -tol:
                    print(f"Warning: rad2 significantly negative at i={i}, j={j}: {rad2}")

                A1 = np.sqrt(max(rad1, 0.0))
                A2 = np.sqrt(max(rad2, 0.0))

                Un_d += Dy[i] * (A1 + A2)

            Un[i] = Un_d

        print("Gamma_0 =", Gamma_0)
        print("Gamma_c =", Gamma_c)
        print("tau_P   =", tau_P)

        return K*tau_P*eVtoGPa, Un

    def get_alpha(DEp_b, b, R, tauP_fixed, branch="negative"):
        """
        Solve alpha from tauP_fixed = tau_P(alpha).

        Parameters
        ----------
        DEp_b : float
            Peierls barrier times b
        b : float
            Burgers vector
        R : float
            Gamma_c / Gamma_0
        tauP_fixed : float
            Target tau_P value
        branch : str
            "negative", "positive", or "both"

        Returns
        -------
        float or tuple
            alpha value, or (alpha_pos, alpha_neg) if branch="both"
        """

        a_b = 2 * b / np.sqrt(3)
        a = np.sqrt(2 / 3) * a_b

        DEp = DEp_b / b
        Gamma_0 = DEp / (R - 1)
        Gamma_c = Gamma_0 * R

        def tauP_from_alpha(alpha):
            alpha_abs = abs(alpha)

            if alpha_abs < 1e-14:
                return np.pi * (Gamma_c - Gamma_0) / (a * b)

            return (
                np.pi * (Gamma_c - Gamma_0)
                / (16 * alpha_abs * a * b)
                * (3 + np.sqrt(1 + 8 * alpha_abs**2))
                * np.sqrt(8 * alpha_abs**2 - 2 + 2 * np.sqrt(1 + 8 * alpha_abs**2))
            )

        tau0 = tauP_from_alpha(0.0)

        if np.isclose(tauP_fixed, tau0, rtol=1e-12, atol=1e-15):
            alpha_abs = 0.0
        else:
            def f(alpha_abs):
                return tauP_from_alpha(alpha_abs) - tauP_fixed

            if tauP_fixed < tau0:
                raise ValueError(
                    f"No real solution: tauP_fixed ({tauP_fixed}) is smaller than tauP(alpha=0) ({tau0})."
                )

            lo = 1e-12
            hi = 1.0
            while f(hi) < 0:
                hi *= 2
                if hi > 1e6:
                    raise RuntimeError("Could not bracket the root for |alpha|.")

            sol = root_scalar(f, bracket=[lo, hi], method="brentq")

            if not sol.converged:
                raise RuntimeError("Root finding failed.")

            alpha_abs = sol.root

        if branch == "negative":
            return -alpha_abs
        elif branch == "positive":
            return alpha_abs
        elif branch == "both":
            return alpha_abs, -alpha_abs
        else:
            raise ValueError("branch must be 'negative', 'positive', or 'both'")

    def get_R(Ek, DEp_b, b, tau_P_md, R_low=1.0001, R_high=1.05, Nsteps_k=10000):
        a_b = 2 * b / np.sqrt(3)
        a = np.sqrt(2 / 3) * a_b
        DEp = DEp_b / b

        def predicted_Ek(R):
            alpha = get_alpha(DEp_b, b, R, tau_P_md, branch="negative")
            Gamma_0 = DEp / (R - 1)
            Gamma_c = Gamma_0 * R

            Dy_k = a / Nsteps_k
            Uk = 0.0

            for j in range(Nsteps_k):
                y01_k = -a / 2 + j * Dy_k
                y02_k = -a / 2 + (j + 1) * Dy_k

                Gamma_y01_k = gamma_func(y01_k, Gamma_c, Gamma_0, alpha, a)
                Gamma_y02_k = gamma_func(y02_k, Gamma_c, Gamma_0, alpha, a)

                rad1 = max((Gamma_y01_k / Gamma_0) ** 2 - 1, 0.0)
                rad2 = max((Gamma_y02_k / Gamma_0) ** 2 - 1, 0.0)

                Uk += Gamma_0 / 2 * Dy_k * (np.sqrt(rad1) + np.sqrt(rad2))

            return 2 * Uk

        def f(R):
            return predicted_Ek(R) - Ek

        sol = root_scalar(f, bracket=[R_low, R_high], method="brentq")

        if not sol.converged:
            raise RuntimeError("Failed to find R")

        R = sol.root
        alpha = get_alpha(DEp_b, b, R, tau_P_md, branch="negative")

        print("--------------------------------------------")
        print("Found R:", R)
        print("Found alpha:", alpha)
        print("Predicted kink pair formation energy:", predicted_Ek(R))
        print("MD kink pair formation energy:", Ek)
        print("Difference:", predicted_Ek(R) - Ek)

        return R, alpha

    def plot_LT_model(Ek, DEp_b, a0, tauP_fixed):
    
        eVtoGPa = 160.21766208
        b = a0 * np.sqrt(3) / 2  # burgers vector
        tau_P_md = tauP_fixed * (1/eVtoGPa)
    
        R, alpha = get_R(Ek, DEp_b, b, tau_P_md, R_low=1.00001, R_high=1.03)
        LT_tau, LT_E = LT_model(DEp_b, R, b, alpha)
    
        plt.figure(figsize=(5, 3))
        plt.plot(LT_tau, LT_E, color='orange', linewidth=2, label='LT model')
        plt.xlabel('τ, GPa', fontsize=18)
        plt.ylabel('ΔH*(τ), eV', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
    return (plot_LT_model,)


@app.cell
def _(mo):
    mo.md("""
    ## Data for Niobium (Nb) at 0K, from MD simulations:
    """)
    return


@app.cell
def _(plot_LT_model):
    # add input
    Ek_Nb = 0.72 # Kink pair formation energy in eV
    DEp_b_Nb = 0.0449 # Peierls energy barrier in eV
    a0_Nb = 3.308165 # equilibrium lattice constant in Angstrom
    tauP_fixed_Nb = 1.375 # Peierls stress in GPa

    plot_LT_model(Ek_Nb, DEp_b_Nb, a0_Nb, tauP_fixed_Nb)
    return DEp_b_Nb, Ek_Nb, a0_Nb, tauP_fixed_Nb


@app.cell
def _(mo):
    mo.md("""
    ## Data for Tungsten (W) at 0K, from MD simulations:
    """)
    return


@app.cell
def _(DEp_b_Nb, Ek_Nb, a0_Nb, plot_LT_model, tauP_fixed_Nb):
    Ek_W = 1.488  # Kink pair formation energy in eV
    DEp_b_W = 0.0986 # Peierls energy barrier in eV
    a0_W = 3.186619 # equilibrium lattice constant in Angstrom
    tauP_fixed_W = 2.875 # Peierls stress in GPa

    plot_LT_model(Ek_Nb, DEp_b_Nb, a0_Nb, tauP_fixed_Nb)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data for molybdenum (Mo) at 0K, from MD simulations:
    """)
    return


@app.cell
def _(plot_LT_model):
    Ek_Mo = 0.989  # Kink pair formation energy in eV
    DEp_b_Mo = 0.0576 # Peierls energy barrier in eV
    a0_Mo = 3.163423 # equilibrium lattice constant in Angstrom
    tauP_fixed_Mo = 1.525 # Peierls stress in GPa

    plot_LT_model(Ek_Mo, DEp_b_Mo, a0_Mo, tauP_fixed_Mo)
    return


if __name__ == "__main__":
    app.run()
