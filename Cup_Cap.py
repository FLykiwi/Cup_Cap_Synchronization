"""
cupcap.py - self-consistent theory of Cup and Cap Synchronization
=================================================================

Companion code to
    R. Wang, Y. Zhang and G. Bianconi,
    "Cup and cap synchronization: higher-order node dynamics from topology".

On a fully connected network (N -> infinity) the dynamics reduces to
    dtheta_i/dt = omega_i - Im[Lambda exp(i theta_i)],
with the real effective field Lambda fixed by the self-consistent equation

    Lambda = F(Lambda) = sigma1 * R_1 + sigma2 * T_2 + sigma3 * T_3,

    T_2 = Theta^(2) / 2^h1,    T_3 = Theta^(3) / 2^(h1+h2-2)

(main text Eqs. 32-33, SI Eqs. 14-16). The order parameters R_{p/2}(Lambda)
are given in SI Eqs. 17 and 21. M = 2 corresponds to sigma3 = 0.

Main entry points
-----------------
CupCap               model definition (h1, h2, sigma3, g, eta)
follow_transition    order parameter R_1 along synchronization/desynchronization
phase_diagram        pitchfork line, saddle-node curve, cusp and grey points
plot_transition, plot_phase_diagram   plotting helpers
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

QUAD_LIMIT = 200

# Colours used in the manuscript for the bias eta (or 1 - eta)
ETA_COLORS = {1.0: "#8c510a", 0.9: "purple", 0.8: "red",
              0.7: "green", 0.6: "orange", 0.5: "blue"}
REGION_COLORS = {"incoherent": "#d6e6f5", "synchronized": "#d9f0d3", "bistable": "#ead9c6"}


# ---------------------------------------------------------------------------
# Intrinsic frequency distributions (symmetric, zero mean)
# ---------------------------------------------------------------------------
def g_gauss(w):
    """Standard normal distribution N(0, 1)."""
    return np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi)


def g_lor(w):
    """Lorentzian distribution with unit width (Delta = 1)."""
    return (1.0 / np.pi) / (w * w + 1.0)


# ---------------------------------------------------------------------------
# Order parameters R_{p/2}(Lambda)
# ---------------------------------------------------------------------------
def _S(w, Lam):
    return np.sqrt(np.abs(1.0 - (w / Lam) ** 2))


def _locked(f, Lam):
    return quad(f, -Lam, Lam, points=[0.0], limit=QUAD_LIMIT)[0]


def _R2_drift(Lam, g):
    """Contribution of drifting oscillators (|w| > Lambda) to R_2.

    Uses 1 - 2x^2 + 2x sqrt(x^2 - 1) = -1/(x + sqrt(x^2 - 1))^2 (no cancellation),
    and geometric breakpoints so that small Lambda is integrated accurately.
    """
    f = lambda w: -g(w) / (abs(w / Lam) + np.sqrt((w / Lam) ** 2 - 1.0)) ** 2
    top = max(100.0, 100.0 * Lam)
    edges = np.geomspace(Lam, top, int(np.ceil(np.log10(top / Lam))) + 2)
    total = sum(quad(f, a, b, limit=QUAD_LIMIT)[0] for a, b in zip(edges[:-1], edges[1:]))
    return 2.0 * (total + quad(f, top, np.inf, limit=QUAD_LIMIT)[0])


@lru_cache(maxsize=2 ** 20)
def _R_unbiased(p, Lam, g):
    """R_{p/2} for p = 1..6; half-integer ones (p odd) are given for eta = 1."""
    if Lam <= 0.0:
        return 0.0
    S = lambda w: _S(w, Lam)
    half = lambda w: np.sqrt(0.5 * (1.0 + S(w)))
    if p == 1:   # X_{1/2}
        return _locked(lambda w: g(w) * half(w), Lam)
    if p == 2:   # R_1
        return _locked(lambda w: g(w) * S(w), Lam)
    if p == 3:   # X_{3/2}
        return _locked(lambda w: g(w) * (2.0 * S(w) - 1.0) * half(w), Lam)
    if p == 4:   # R_2
        return _locked(lambda w: g(w) * (1.0 - 2.0 * (w / Lam) ** 2), Lam) + _R2_drift(Lam, g)
    if p == 5:   # X_{5/2}
        return _locked(lambda w: g(w) * (3.0 - 4.0 * (w / Lam) ** 2 - 2.0 * S(w)) * half(w), Lam)
    if p == 6:   # R_3
        return _locked(lambda w: g(w) * (1.0 - 4.0 * (w / Lam) ** 2) * S(w), Lam)
    raise ValueError(f"R_{{{p}/2}} not implemented (|p| <= 6, i.e. h1, h2 <= 5)")


def order_parameter(p, Lam, g, eta=0.5):
    """R_{p/2}(Lambda): integer p -> R_{p/2}; odd p -> half-integer X_{p/2}.

    Half-integer order parameters carry the factor (2 eta - 1), where eta is the
    bias (fraction of phases in (-pi, pi]). X_{-p/2} = X_{p/2} (real).
    """
    p = abs(int(p))
    if p == 0:
        return 1.0
    value = _R_unbiased(p, float(Lam), g)
    return (2.0 * eta - 1.0) * value if p % 2 else value


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CupCap:
    """Cup and Cap Synchronization on a fully connected network.

    h1, h2 : exponents of f_r(x) = cos(x)^(h_r - 1)
    sigma3 : four-body coupling (sigma3 = 0 gives the M = 2 model)
    g      : intrinsic frequency distribution
    eta    : bias of the initial synchronized state (only matters if h1 or h2 is even)
    """
    h1: int
    h2: int = 1
    sigma3: float = 0.0
    g: object = g_gauss
    eta: float = 0.5

    @property
    def M(self):
        return 3 if self.sigma3 != 0 else 2

    @property
    def biased(self):
        """True if half-integer order parameters enter F (the bias eta matters)."""
        return self.h1 % 2 == 0 or (self.M == 3 and self.h2 % 2 == 0)

    def _R(self, p, Lam):
        return order_parameter(p, Lam, self.g, self.eta)

    def components(self, Lam):
        """(R_1, T_2, T_3) such that F = sigma1 R_1 + sigma2 T_2 + sigma3 T_3."""
        h1, h2, R = self.h1, self.h2, self._R
        R1 = R(2, Lam)
        theta2 = sum(comb(h1, n) * R(2 * n - h1 + 1, Lam) * R(2 * n - h1 - 1, Lam)
                     for n in range(h1 + 1))
        T3 = 0.0
        if self.sigma3 != 0:
            theta3 = sum(comb(h1 - 1, n) * comb(h2 - 1, m)
                         * R(1 - 2 * n + h1, Lam) * R(2 * n - 2 * m - h1 + h2, Lam)
                         * R(2 * m - h2 + 1, Lam)
                         for n in range(h1) for m in range(h2))
            T3 = theta3 / 2 ** (h1 + h2 - 2)
        return np.array([R1, theta2 / 2 ** h1, T3])

    def F(self, Lam, sigma1, sigma2):
        """Right-hand side of the self-consistent equation Lambda = F(Lambda)."""
        return np.dot([sigma1, sigma2, self.sigma3], self.components(Lam))

    def R1(self, Lam):
        return self._R(2, Lam)


# ---------------------------------------------------------------------------
# Following the transitions
# ---------------------------------------------------------------------------
def solve(model, sigma1, sigma2, Lam0, k=0.5, max_iter=2000, tol=1e-6):
    """Stable solution of Lambda = F(Lambda) reached from Lam0 by damped iteration
    Lambda <- max(Lambda + k (F(Lambda) - Lambda), 0). Only stable fixed points attract."""
    Lam = Lam0
    for _ in range(max_iter):
        d = model.F(Lam, sigma1, sigma2) - Lam
        Lam = max(Lam + k * d, 0.0)
        if abs(d) < tol:
            break
    return Lam


def follow_transition(model, sigma1_values, sigma2, sync=True, desync=True,
                      Lam_perturbation=0.01, Lam_sync=50.0):
    """Order parameter R_1 along an adiabatic sweep of sigma1.

    sync   : increase sigma1 starting from the incoherent state (unbiased, eta = 0.5)
    desync : decrease sigma1 starting from a synchronized state with bias model.eta

    Returns a dict {"sync": array, "desync": array} ordered like sigma1_values.
    """
    sigma1_values = np.asarray(sigma1_values)
    out = {}
    if sync:
        unbiased = replace(model, eta=0.5)
        Lam, R = 0.0, []
        for s1 in sigma1_values:
            Lam = solve(unbiased, s1, sigma2, max(Lam, Lam_perturbation))
            R.append(unbiased.R1(Lam))
        out["sync"] = np.array(R)
    if desync:
        Lam, R = Lam_sync, []
        for s1 in sigma1_values[::-1]:
            Lam = solve(model, s1, sigma2, Lam)
            R.append(model.R1(Lam))
        out["desync"] = np.array(R[::-1])
    return out


# ---------------------------------------------------------------------------
# Phase diagram in the (sigma1, sigma2) plane
# ---------------------------------------------------------------------------
@dataclass
class PhaseDiagram:
    model: CupCap
    d1: np.ndarray            # dF/dLambda(0) per component
    pitchfork_sub: np.ndarray  # (n, 2) points (sigma1, sigma2)
    pitchfork_sup: np.ndarray
    saddle_node: np.ndarray
    cusp: tuple = None
    grey: tuple = None

    def sigma2_pitchfork(self, sigma1):
        d1, s3 = self.d1, self.model.sigma3
        return (1 - d1[0] * np.asarray(sigma1) - d1[2] * s3) / d1[1]


def _derivatives_at_zero(model, eps):
    """Finite-difference F'(0), F''(0), F'''(0) per component (F(0) = 0)."""
    c1, c2, c3 = (model.components(n * eps) for n in (1, 2, 3))
    d1 = (4 * c1 - c2) / (2 * eps)
    d2 = (-5 * c1 + 4 * c2 - c3) / eps ** 2
    d3 = (3 * c1 - 3 * c2 + c3) / eps ** 3
    return d1, d2, d3


def phase_diagram(model, sigma1_range=(-10, 10), n_points=2001,
                  Lambda_star=np.geomspace(1e-2, 65, 200), eps=1e-3, tol=1e-6):
    """Bifurcation structure of Lambda = F(Lambda) (SI Sec. III).

    * pitchfork line F'(0) = 1, subcritical where F''(0) > 0
      (or F'''(0) > 0 when F''(0) vanishes identically, e.g. eta = 0.5);
    * saddle-node curve F(L*) = L*, F'(L*) = 1, traced by sweeping L* > 0;
    * cusp: where the pitchfork line turns from sub- to supercritical;
    * grey point: biased case, F'''(0) = 0 on a fully subcritical line.
    """
    s3 = model.sigma3
    d1, d2, d3 = _derivatives_at_zero(model, eps)
    s1 = np.linspace(*sigma1_range, n_points)
    s2 = (1 - d1[0] * s1 - d1[2] * s3) / d1[1]

    along = lambda d: s1 * d[0] + s2 * d[1] + s3 * d[2]      # a derivative on the line
    generic = abs(d2[1]) > tol or abs(s3 * d2[2]) > tol      # F''(0) not identically 0
    curvature = along(d2) if generic else along(d3)
    sub = curvature >= 0

    def zero_crossing(values):
        idx = np.where(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        t = values[i] / (values[i] - values[i + 1])
        return (s1[i] + t * (s1[i + 1] - s1[i]), s2[i] + t * (s2[i + 1] - s2[i]))

    cusp = zero_crossing(curvature)
    grey = zero_crossing(along(d3)) if (generic and sub.all()) else None

    # Saddle-node curve: F is linear in (sigma1, sigma2), so for each L* the two
    # conditions form a 2x2 linear system.
    sn, det_old = [], 0.0
    for L in Lambda_star:
        c = model.components(L)
        dc = (model.components(L + eps) - model.components(L - eps)) / (2 * eps)
        det = c[0] * dc[1] - dc[0] * c[1]
        if det * det_old < 0:
            break
        rhs0, rhs1 = L - s3 * c[2], 1 - s3 * dc[2]
        sn.append(((rhs0 * dc[1] - rhs1 * c[1]) / det, (c[0] * rhs1 - dc[0] * rhs0) / det))
        det_old = det

    return PhaseDiagram(model, d1, np.column_stack([s1[sub], s2[sub]]),
                        np.column_stack([s1[~sub], s2[~sub]]), np.array(sn), cusp, grey)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_transition(ax, sigma1, curves, xlim=None, title=None):
    """curves: list of (R_values, label, color, linestyle)."""
    for values, label, color, ls in curves:
        ax.plot(sigma1, values, color=color, ls=ls, lw=2, label=label)
    ax.set_xlabel(r"$\sigma_1$")
    ax.set_ylabel(r"$R_1$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)


def plot_phase_diagram(ax, pd, xlim=(-10, 10), ylim=(-10, 10), shade=True,
                       sn_color="#8c510a", sn_label="saddle-node", annotate=True,
                       show_pitchfork=True):
    """Draw a PhaseDiagram; shading shows incoherent / synchronized / bistable regions."""
    if shade:
        X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
        d1, s3 = pd.d1, pd.model.sigma3
        sync = X * d1[0] + Y * d1[1] + s3 * d1[2] > 1          # incoherent state unstable
        region = np.where(sync, 1, 0)
        if len(pd.saddle_node) > 1:
            sn = pd.saddle_node[np.argsort(pd.saddle_node[:, 1])]
            inside = (Y >= sn[0, 1]) & (Y <= sn[-1, 1])
            right_of_sn = X > np.interp(Y, sn[:, 1], sn[:, 0])
            region = np.where(~sync & inside & right_of_sn, 2, region)
        cmap = plt.matplotlib.colors.ListedColormap(
            [REGION_COLORS["incoherent"], REGION_COLORS["synchronized"], REGION_COLORS["bistable"]])
        ax.pcolormesh(X, Y, region, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto",
                      rasterized=True)
        for name, color in REGION_COLORS.items():
            ax.fill([], [], color=color, label=name)

    if show_pitchfork and len(pd.pitchfork_sub):
        ax.plot(*pd.pitchfork_sub.T, color="blue", lw=2, zorder=4, label="pitchfork (subcritical)")
    if show_pitchfork and len(pd.pitchfork_sup):
        ax.plot(*pd.pitchfork_sup.T, color="blue", lw=2, ls="--", zorder=4,
                label="pitchfork (supercritical)")
    if len(pd.saddle_node):
        ax.plot(*pd.saddle_node.T, color=sn_color, lw=2, label=sn_label)
    for point, color, name in ((pd.cusp, "black", "cusp"), (pd.grey, "grey", None)):
        if point is not None:
            ax.plot(*point, "o", color=color, ms=8, zorder=5, label=name)
            if annotate:
                ax.annotate(f"({point[0]:.2f}, {point[1]:.2f})", point, xytext=(8, 6),
                            textcoords="offset points", fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.set_xlabel(r"$\sigma_1$")
    ax.set_ylabel(r"$\sigma_2$")
