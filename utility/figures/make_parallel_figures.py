r"""Module: utility.figures.make_parallel_figures

Description: Generates the figures used by PARALLEL_GUIDE.md into doc/figures/. Figure 2 is
    computed from the real mesh and partitioner. Figure 5 plots the measured strong-scaling
    numbers recorded in ARCHITECTURE.md. Figures 1, 3 and 4 are schematics. Colours follow the
    validated reference palette.

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.mesh import Mesh  # noqa: E402
from src.partition import _ring, partition_rcb  # noqa: E402

OUT = os.path.join(ROOT, "doc", "figures")

# --- design tokens (light surface) -------------------------------------------
SURFACE = "#fcfcfb"
PANEL = "#f4f3ef"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RANK_COLORS = [BLUE, ORANGE, AQUA, YELLOW]

ROLE = {  # what a step *is*, in diagrams
    "compute": BLUE,
    "comm": ORANGE,
    "reduce": AQUA,
    "io": MUTED,
}
ROLE_NAME = {
    "compute": "Local computation",
    "comm": "Point-to-point communication",
    "reduce": "Collective operation",
    "io": "Output / diagnostics",
}

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
    }
)


def _blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def _legend(fig, handles, ncol, y=0.035):
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=9.5,
        handlelength=1.4,
        columnspacing=2.2,
        bbox_to_anchor=(0.5, y),
    )
    for t in leg.get_texts():
        t.set_color(INK2)


def _title(ax, x, y, title, sub):
    ax.text(x, y, title, fontsize=15.5, fontweight="semibold", color=INK, va="top")
    ax.text(x, y - 3.4, sub, fontsize=9.8, color=INK2, va="top")


def _card(ax, x, top, w, h, title, sub, role):
    """White card with a thin role-coloured accent bar; text stays in ink."""
    ax.add_patch(
        FancyBboxPatch(
            (x, top - h),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.0",
            fc="white",
            ec=AXIS,
            lw=0.9,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle((x + 0.7, top - h + 0.9), 0.9, h - 1.8, fc=ROLE[role], ec="none", zorder=4)
    )
    ax.text(
        x + 3.0,
        top - h * 0.34,
        title,
        fontsize=10.6,
        fontweight="semibold",
        va="center",
        color=INK,
        zorder=5,
    )
    ax.text(x + 3.0, top - h * 0.70, sub, fontsize=8.4, va="center", color=INK2, zorder=5)


def _arrow(ax, a, b, rad=0.0, color=MUTED, lw=1.3, style="-|>", ms=9):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=style,
            mutation_scale=ms,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )


# =============================================================================
# Figure 1 - workflow
# =============================================================================
def fig_workflow():
    fig = plt.figure(figsize=(12.4, 7.9))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _blank(ax, (0, 124), (-5.5, 74))
    _title(
        ax,
        2,
        72.5,
        "Parallel finite-volume workflow",
        "Every rank runs the same program on its own subdomain; "
        "communication happens only where marked.",
    )

    ax.text(2, 59.5, "SETUP", fontsize=10, fontweight="bold", color=INK2)
    ax.text(11.2, 59.5, "once, before the time loop", fontsize=10, color=MUTED)
    ax.text(66, 59.5, "TIME LOOP", fontsize=10, fontweight="bold", color=INK2)
    ax.text(79.5, 59.5, "repeated every step", fontsize=10, color=MUTED)

    setup = [
        (
            "Read and analyse the global mesh",
            "geometry, connectivity; identical on every rank",
            "compute",
        ),
        ("Partition the cells", "deterministic rule: cell → owning rank", "compute"),
        (
            "Build owned set and halo rings",
            "halo depth set by the scheme's stencil reach",
            "compute",
        ),
        (
            "Re-index to local numbering",
            "owned + halo cells; slice geometry and boundary data",
            "compute",
        ),
        (
            "Handshake: who sends what to whom",
            "all-to-all of 'cells I need' lists → send / receive plans",
            "comm",
        ),
        ("Initialise the local state", "slice the initial condition onto local cells", "compute"),
    ]
    top0, step, h = 56.0, 8.9, 7.2
    for i, (t, s, r) in enumerate(setup):
        top = top0 - i * step
        _card(ax, 2, top, 52, h, t, s, r)
        if i < len(setup) - 1:
            _arrow(ax, (28, top - h), (28, top - step), lw=1.2, ms=8)

    # container for the loop
    ax.add_patch(
        FancyBboxPatch(
            (62, 2.2),
            60.5,
            54.8,
            boxstyle="round,pad=0,rounding_size=1.6",
            fc=PANEL,
            ec=GRID,
            lw=1.0,
            zorder=1,
        )
    )
    loop = [
        ("Global time step", "local stability limit → all-reduce (min)", "reduce"),
        ("Halo exchange", "refresh ghost cells of the state (non-blocking)", "comm"),
        ("Reconstruction", "gradients and limiters on owned + inner halo", "compute"),
        ("Face fluxes", "Riemann solver, once per face touching an owned cell", "compute"),
        ("Residual and update", "assemble per owned cell; advance one stage", "compute"),
        ("Monitor / output (every N steps)", "reduce norms; gather to rank 0", "io"),
    ]
    x0, w = 70, 43
    tops = [top0 - 1 - i * step for i in range(len(loop))]
    for (t, s, r), top in zip(loop, tops):
        _card(ax, x0, top, w, h, t, s, r)
    for i in range(len(loop) - 1):
        _arrow(ax, (x0 + w / 2, tops[i] - h), (x0 + w / 2, tops[i + 1]), lw=1.2, ms=8)

    # loops, routed outside the cards: R-K stages (card 5 -> 2) and next step (card 6 -> 1)
    cy = [t - h / 2 for t in tops]

    def route(x_from, ya, yb, dx, color):
        xr = x_from + dx
        ax.plot(
            [x_from, xr, xr], [ya, ya, yb], color=color, lw=1.5, zorder=2, solid_joinstyle="round"
        )
        _arrow(ax, (xr, yb), (x_from, yb), color=color, lw=1.5, ms=9)

    route(x0 + w, cy[4], cy[1], 3.4, ORANGE)
    ax.text(
        x0 + w + 6.1,
        (cy[1] + cy[4]) / 2,
        "next Runge–Kutta stage",
        rotation=90,
        fontsize=8.8,
        color=INK2,
        ha="center",
        va="center",
    )
    route(x0, cy[5], cy[0], -3.4, AQUA)
    ax.text(
        x0 - 6.1,
        (cy[0] + cy[5]) / 2,
        "next time step",
        rotation=90,
        fontsize=8.8,
        color=INK2,
        ha="center",
        va="center",
    )

    # setup -> loop
    _arrow(
        ax, (54, top0 - 5 * step - h / 2), (62, top0 - 5 * step - h / 2), lw=1.6, color=INK2, ms=10
    )

    _legend(fig, [Patch(fc=ROLE[k], ec="none", label=ROLE_NAME[k]) for k in ROLE], 4)
    return fig


# =============================================================================
# Figure 2 - partitioned mesh (real data)
# =============================================================================
def _mesh_polys(mesh):
    idx = mesh.node_tag_map[mesh.elem_conn]
    return mesh.node_coords[idx][:, :, :2]


def fig_partition(n=14, nparts=4, focus=0):
    mesh = Mesh.unit_square(n)
    geo = mesh.fvm_geometry()
    part = partition_rcb(geo.centroids, nparts)
    polys = _mesh_polys(mesh)

    mine = part == focus
    owned = np.flatnonzero(mine)
    seen = mine.copy()
    ring1 = _ring(geo.neighbors, owned, seen)
    seen[ring1] = True
    ring2 = _ring(geo.neighbors, ring1, seen)
    cls = np.full(mesh.nelem, 3)  # 3 = not stored
    cls[owned], cls[ring1], cls[ring2] = 0, 1, 2

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.14, wspace=0.05)
    fig.text(
        0.02,
        0.965,
        "Domain decomposition on a real mesh",
        fontsize=15.5,
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.02,
        0.915,
        f"{mesh.nelem} triangles split over {nparts} ranks by recursive coordinate "
        "bisection; right: what one rank actually stores.",
        fontsize=9.8,
        color=INK2,
        va="top",
    )

    # (a) all ranks
    ax = axs[0]
    cols = [RANK_COLORS[p % len(RANK_COLORS)] for p in part]
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=SURFACE, linewidths=0.5))
    # cut faces: thick line between different parts
    segs = []
    for i in range(mesh.nelem):
        for j in range(geo.neighbors.shape[1]):
            nb = geo.neighbors[i, j]
            if nb > i and part[nb] != part[i]:
                nodes = mesh.node_tag_map[mesh.elem_faces[i, j]]
                segs.append(mesh.node_coords[nodes][:, :2])
    for s in segs:
        ax.plot(s[:, 0], s[:, 1], color=INK, lw=2.0, solid_capstyle="round")
    for p in range(nparts):
        c = geo.centroids[part == p, :2].mean(axis=0)
        ax.text(
            c[0],
            c[1],
            f"Rank {p}",
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="semibold",
            color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.92),
        )
    _blank(ax, (-0.02, 1.02), (-0.02, 1.02))
    ax.set_title("(a)  Global mesh, partitioned", loc="left", fontsize=11.5, color=INK, pad=8)
    ax.legend(
        handles=[Line2D([0], [0], color=INK, lw=2.0, label="cut faces (between ranks)")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.0),
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
    )

    # (b) rank `focus` local view
    ax = axs[1]
    base = RANK_COLORS[focus % len(RANK_COLORS)]
    tint = {0: (base, 1.0), 1: (base, 0.50), 2: (base, 0.22)}
    cols = []
    for c in cls:
        if c == 3:
            cols.append("#f0efec")
        else:
            cols.append(matplotlib.colors.to_rgba(*tint[c]))
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=SURFACE, linewidths=0.5))
    # outline owned region
    for cell in owned:
        i = int(cell)
        for j in range(geo.neighbors.shape[1]):
            nb = geo.neighbors[i, j]
            if nb == -1 or part[nb] != focus:
                nodes = mesh.node_tag_map[mesh.elem_faces[i, j]]
                s = mesh.node_coords[nodes][:, :2]
                ax.plot(s[:, 0], s[:, 1], color=INK, lw=2.0, solid_capstyle="round")
    _blank(ax, (-0.02, 1.02), (-0.02, 1.02))
    ax.set_title(f"(b)  What rank {focus} stores", loc="left", fontsize=11.5, color=INK, pad=8)
    ax.legend(
        handles=[
            Patch(
                fc=matplotlib.colors.to_rgba(base, 1.0), ec="none", label=f"owned ({len(owned)})"
            ),
            Patch(
                fc=matplotlib.colors.to_rgba(base, 0.5),
                ec="none",
                label=f"halo ring 1 ({len(ring1)})",
            ),
            Patch(
                fc=matplotlib.colors.to_rgba(base, 0.22),
                ec="none",
                label=f"halo ring 2 ({len(ring2)})",
            ),
            Patch(fc="#f0efec", ec="none", label="not stored"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.0),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
        columnspacing=1.6,
    )
    return fig


# =============================================================================
# Figure 3 - how the halo depth follows from the stencil
# =============================================================================
def fig_halo_depth():
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.15)
    fig.text(
        0.02,
        0.965,
        "Halo depth follows from the stencil",
        fontsize=15.5,
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.02,
        0.885,
        "Each dependent stage reads one neighbour ring further out, so the region where "
        "results are valid shrinks by one ring per stage.",
        fontsize=9.8,
        color=INK2,
        va="top",
    )

    ncell, o0, o1 = 14, 4, 9  # owned cells o0..o1 (inclusive); neighbours belong to other ranks
    cw, ch, gap = 1.0, 1.0, 0.12

    def draw(ax, title, rows, depth):
        # rows: list of (label, valid_range); drawn bottom-up with the last row at the top
        nrows = len(rows)
        _blank(ax, (-7.2, ncell * (cw + gap) + 0.4), (-2.6, 6.6))
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=6)
        for r, (label, (a, b)) in enumerate(rows):
            y = r * 2.0
            for i in range(ncell):
                if a <= i <= b:
                    d = 0 if o0 <= i <= o1 else min(abs(i - o0), abs(i - o1))
                    alpha = {0: 1.0, 1: 0.5, 2: 0.22}.get(d, 0.22)
                    fc: str | tuple = matplotlib.colors.to_rgba(BLUE, alpha)
                    ec = "none"
                else:
                    fc, ec = "none", GRID
                ax.add_patch(Rectangle((i * (cw + gap), y), cw, ch, fc=fc, ec=ec, lw=0.9, zorder=3))
            ax.text(-0.4, y + ch / 2, label, ha="right", va="center", fontsize=9.3, color=INK2)
        # dependency hints: arrows from row r to r+1 at the ends of the valid range
        for r in range(1, nrows):
            a, b = rows[r][1]
            for i, sgn in ((a, -1), (b, 1)):
                x_top = i * (cw + gap) + cw / 2
                x_bot = (i + sgn) * (cw + gap) + cw / 2
                _arrow(ax, (x_bot, (r - 1) * 2.0 + ch), (x_top, r * 2.0), color=INK2, lw=1.0, ms=7)
        # footer: ownership
        y0 = -0.9
        xo0, xo1 = o0 * (cw + gap), o1 * (cw + gap) + cw
        ax.plot([xo0, xo1], [y0, y0], color=INK, lw=2.0, solid_capstyle="butt")
        ax.text(
            (xo0 + xo1) / 2,
            y0 - 0.7,
            "owned by this rank",
            ha="center",
            va="top",
            fontsize=9.3,
            color=INK,
        )
        for xs, txt in ((0, "other rank"), (o1 + 1 + 0.9, "other rank")):
            xl = (o1 + 1) * (cw + gap) + 0.1 if xs else 0
            xr = ncell * (cw + gap) if xs else o0 * (cw + gap) - 0.1
            ax.plot([xl, xr], [y0, y0], color=MUTED, lw=2.0, solid_capstyle="butt")
            ax.text((xl + xr) / 2, y0 - 0.7, txt, ha="center", va="top", fontsize=9.3, color=MUTED)
        ax.text(
            ncell * (cw + gap) / 2 + 0.2,
            5.9,
            f"→ halo depth {depth}",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold",
            color=INK,
        )

    d1 = o0 - 1
    d1b = o1 + 1
    draw(
        axs[0],
        "(a)  First-order: flux uses neighbour states",
        [("① state U  (after exchange)", (d1, d1b)), ("② fluxes → residual", (o0, o1))],
        depth=1,
    )
    d2, d2b = o0 - 2, o1 + 2
    draw(
        axs[1],
        "(b)  Second-order: flux uses neighbour gradients",
        [
            ("① state U  (after exchange)", (d2, d2b)),
            ("② gradients, limiters", (o0 - 1, o1 + 1)),
            ("③ fluxes → residual", (o0, o1)),
        ],
        depth=2,
    )
    _legend(
        fig,
        [
            Patch(fc=matplotlib.colors.to_rgba(BLUE, 1.0), ec="none", label="owned cell"),
            Patch(fc=matplotlib.colors.to_rgba(BLUE, 0.5), ec="none", label="halo ring 1"),
            Patch(fc=matplotlib.colors.to_rgba(BLUE, 0.22), ec="none", label="halo ring 2"),
            Patch(fc="none", ec=GRID, label="not valid at this stage"),
        ],
        4,
        y=0.02,
    )
    return fig


# =============================================================================
# Figure 4 - one time step on three ranks (schematic)
# =============================================================================
def fig_timeline():
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    _blank(ax, (0, 124), (0, 56))
    _title(
        ax,
        2,
        55,
        "One time step on three ranks",
        "Schematic, not to scale. The slowest neighbour sets the pace at each exchange, "
        "so load balance matters.",
    )

    lanes_y = {0: 33.0, 1: 22.0, 2: 11.0}
    lane_h = 6.4
    t0 = 12.0
    scale = 7.3  # units per time unit

    reduce_w, exch_w = 0.85, 0.75
    comp = {0: (5.0, 5.0), 1: (5.7, 5.7), 2: (4.5, 4.5)}  # compute time per stage per rank

    def X(t):
        return t0 + t * scale

    for r, y in lanes_y.items():
        ax.text(
            t0 - 1.4,
            y + lane_h / 2,
            f"Rank {r}",
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="semibold",
            color=INK,
        )
        ax.plot([X(0), X(14.6)], [y - 0.9, y - 0.9], color=GRID, lw=0.8, zorder=0)

    def seg(r, ta, tb, role, text=None, fc=None, tcolor="white"):
        y = lanes_y[r]
        ax.add_patch(
            Rectangle(
                (X(ta), y),
                (tb - ta) * scale - 0.35,
                lane_h,
                fc=fc or ROLE[role],
                ec="none",
                zorder=3,
            )
        )
        if text:
            ax.text(
                X(ta) + ((tb - ta) * scale - 0.35) / 2,
                y + lane_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=8.6,
                color=tcolor,
                fontweight="semibold",
                zorder=4,
            )

    t = 0.0
    boundaries = []
    for r in lanes_y:
        seg(r, t, t + reduce_w, "reduce", "")
    t += reduce_w
    boundaries.append(("global time step\n(all-reduce)", 0.0, t))
    for stage in range(2):
        for r in lanes_y:
            seg(r, t, t + exch_w, "comm", "")
        ex_a = t
        t += exch_w
        ends = {}
        for r in lanes_y:
            seg(
                r,
                t,
                t + comp[r][stage],
                "compute",
                "reconstruct · flux · update"
                if comp[r][stage] > 4.9
                else "reconstruct · flux · update",
            )
            ends[r] = t + comp[r][stage]
        tmax = max(ends.values())
        for r in lanes_y:
            if ends[r] < tmax - 1e-6:
                seg(r, ends[r], tmax, "io", "idle", fc="#ecebe6", tcolor=INK2)
        boundaries.append((f"halo exchange + Runge–Kutta stage {stage + 1}", ex_a, tmax))
        # message arrows during the exchange (neighbours 0-1 and 1-2)
        xm = X(ex_a) + (exch_w * scale - 0.35) / 2
        for a, b in ((0, 1), (1, 2)):
            ya, yb = lanes_y[a], lanes_y[b]
            _arrow(ax, (xm - 0.9, yb + lane_h), (xm - 0.9, ya), color=INK, lw=1.1, ms=7)
            _arrow(ax, (xm + 0.9, ya), (xm + 0.9, yb + lane_h), color=INK, lw=1.1, ms=7)
        t = tmax
    for r in lanes_y:
        seg(r, t, t + reduce_w, "reduce", "")
    boundaries.append(("next step", t, t + reduce_w))

    # stage brackets above lanes
    y_bracket = lanes_y[0] + lane_h + 3.4
    for label, t_a, t_b in boundaries:
        x0, x1 = X(t_a), X(t_b) - 0.35
        ax.plot([x0, x1], [y_bracket, y_bracket], color=INK2, lw=1.2)
        ax.plot([x0, x0], [y_bracket - 0.7, y_bracket], color=INK2, lw=1.2)
        ax.plot([x1, x1], [y_bracket - 0.7, y_bracket], color=INK2, lw=1.2)
        ax.text(
            (x0 + x1) / 2,
            y_bracket + 0.7,
            label,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=INK2,
        )

    _legend(
        fig,
        [
            Patch(fc=ROLE["reduce"], ec="none", label="all-reduce (global dt)"),
            Patch(fc=ROLE["comm"], ec="none", label="halo exchange"),
            Patch(fc=ROLE["compute"], ec="none", label="local computation"),
            Patch(fc="#ecebe6", ec="none", label="idle (waiting for neighbours)"),
            Line2D([0], [0], color=INK, lw=1.1, marker=">", markersize=5, label="messages"),
        ],
        5,
        y=0.0,
    )
    return fig


# =============================================================================
# Figure 5 - measured strong scaling
# =============================================================================
def fig_scaling():
    ranks = np.array([1, 2, 4, 8])
    run1 = np.array([100.05, 70.75, 41.73, 29.22])
    run2 = np.array([90.68, 69.18, 41.65, 27.19])
    s1, s2 = run1[0] / run1, run2[0] / run2

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.085, right=0.93, top=0.80, bottom=0.17)
    fig.text(
        0.085, 0.965, "Strong scaling on one node", fontsize=15.5, fontweight="semibold", va="top"
    )
    fig.text(
        0.085,
        0.905,
        "180,000 cells, Roe + RK2, one numba thread per rank, Intel i7-10700K "
        "(8 cores), Microsoft MPI",
        fontsize=9.3,
        color=INK2,
        va="top",
    )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)

    ax.plot(ranks, ranks, color=MUTED, lw=1.4, zorder=2)
    ax.plot(ranks, s1, color=BLUE, lw=2.0, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4)
    ax.plot(ranks, s2, color=ORANGE, lw=2.0, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)

    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks], fontsize=10)
    ax.minorticks_off()
    ax.set_xlim(0.6, 10.9)
    ax.set_ylim(0, 8.6)
    ax.set_yticks(range(0, 9, 2))
    ax.tick_params(axis="both", length=0, labelsize=10)
    ax.set_xlabel("MPI ranks", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Speedup over 1 rank", fontsize=10.5, labelpad=8)

    # selective direct labels at the end of each series
    ax.annotate("ideal", (8, 8), xytext=(8.3, 8.0), fontsize=10, color=INK2, va="center")
    ax.annotate(
        f"{s1[-1]:.2f}×  (run 1)",
        (8, s1[-1]),
        xytext=(8.3, s1[-1] + 0.3),
        fontsize=10,
        color=INK2,
        va="center",
    )
    ax.annotate(
        f"{s2[-1]:.2f}×  (run 2)",
        (8, s2[-1]),
        xytext=(8.3, s2[-1] - 0.5),
        fontsize=10,
        color=INK2,
        va="center",
    )
    ax.text(
        7.9,
        1.0,
        f"parallel efficiency at 8 ranks: {s2[-1] / 8:.0%}–{s1[-1] / 8:.0%}",
        ha="right",
        fontsize=10,
        color=INK2,
    )

    _legend(
        fig,
        [
            Line2D([0], [0], color=MUTED, lw=1.4, label="ideal (linear)"),
            Line2D(
                [0], [0], color=BLUE, lw=2.0, marker="o", ms=7, mec=SURFACE, label="measured, run 1"
            ),
            Line2D(
                [0],
                [0],
                color=ORANGE,
                lw=2.0,
                marker="o",
                ms=7,
                mec=SURFACE,
                label="measured, run 2",
            ),
        ],
        3,
        y=0.015,
    )
    return fig


FIGURES = {
    "fig1_workflow": fig_workflow,
    "fig2_partition": fig_partition,
    "fig3_halo_depth": fig_halo_depth,
    "fig4_timeline": fig_timeline,
    "fig5_scaling": fig_scaling,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in FIGURES.items():
        fig = fn()
        path = os.path.join(OUT, f"{name}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
