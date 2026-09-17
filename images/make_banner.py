"""MILES-TAILS repo banner: FFS trajectory schematic + heavy-tailed distribution.
Pure matplotlib, Okabe-Ito palette, no AI art."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BLUE, VERM, SKY, GRAY, INK = "#0072B2", "#D55E00", "#56B4E9", "#9aa5ae", "#1b2733"
rng = np.random.default_rng(7)

def ou_path(y0, n, drift=-0.004, sigma=0.022, pull=0.012, target=0.08):
    y = np.empty(n); y[0] = y0
    for i in range(1, n):
        y[i] = y[i-1] + drift + pull*(target - y[i-1]) + sigma*rng.standard_normal()
    return y

fig = plt.figure(figsize=(12.0, 5.4), dpi=200)
fig.patch.set_facecolor("white")

# ---------------- title ----------------
fig.text(0.055, 0.90, "MILES-TAILS", fontsize=34, fontweight="bold", color=INK,
         family="DejaVu Sans")
fig.text(0.055, 0.825, "Rare-event sampling for AI weather prediction",
         fontsize=15, color="#5a6a76")
fig.text(0.945, 0.855, "Forward Flux Sampling", fontsize=12, color=GRAY, ha="right", style="italic")

# ---------------- left panel: FFS trajectories ----------------
ax = fig.add_axes([0.055, 0.13, 0.54, 0.60])
interfaces = [0.28, 0.48, 0.68]
target_lo = 0.86
n = 260
x = np.arange(n)

# quiescent basin + target band
ax.axhspan(0.0, 0.14, color=BLUE, alpha=0.08, lw=0)
ax.axhspan(target_lo, 1.06, color=VERM, alpha=0.10, lw=0)
for lam in interfaces:
    ax.axhline(lam, color=GRAY, lw=0.9, ls=(0, (5, 4)), alpha=0.85)
ax.axhline(target_lo, color=VERM, lw=1.1, ls=(0, (5, 3)), alpha=0.9)

# stage-0 failures from the basin
for _ in range(9):
    y = ou_path(0.10, n)
    ax.plot(x, np.clip(y, 0, 1.04), color=GRAY, lw=0.9, alpha=0.45)

def stage(from_y, start_x, n_paths, reach, color, alpha, lw=1.1):
    """n_paths walkers launched at (start_x, from_y); `reach` of them get a
    drift boost so they cross the next level; returns crossing points."""
    crossings = []
    for k in range(n_paths):
        boost = 0.010 if k < reach else 0.0
        m = n - start_x
        y = np.empty(m); y[0] = from_y
        for i in range(1, m):
            y[i] = y[i-1] + boost - 0.003 + 0.012*(0.08 - y[i-1])*(boost == 0) + 0.02*rng.standard_normal()
        seg = np.clip(y, 0, 1.04)
        ax.plot(start_x + np.arange(m), seg, color=color, lw=lw, alpha=alpha)
        crossings.append((start_x + m//3, seg[m//3]))
    return crossings

# staged climbers, hand-tuned for legibility
def climber(x0, y0, x1, y1, wob=0.015, col=SKY, lw=1.3, al=0.9):
    m = x1 - x0
    base = np.linspace(y0, y1, m)
    y = base + wob*np.convolve(rng.standard_normal(m), np.ones(9)/9, mode="same")
    ax.plot(np.arange(x0, x1), np.clip(y, 0, 1.04), color=col, lw=lw, alpha=al)
    return y[-1]

# stochastic excursions that cross one or two interfaces then fall back
def excursion(x0, peak, m):
    t = np.linspace(0, 1, m)
    envelope = peak * np.sin(np.pi * t) ** 1.4 + 0.09
    y = envelope + 0.035*np.convolve(rng.standard_normal(m), np.ones(11)/11, mode="same") \
        + 0.012*rng.standard_normal(m)
    ax.plot(np.arange(x0, x0+m), np.clip(y, 0.01, 1.04), color=BLUE, lw=1.0, alpha=0.5)
for x0, pk, m in [(12, 0.30, 85), (55, 0.46, 95), (115, 0.33, 75), (150, 0.55, 105), (205, 0.40, 55)]:
    excursion(x0, pk, m)

# the single reactive trajectory: basin -> through all interfaces -> genesis
# staged: linger below each interface, then push through (FFS-like)
rng2 = np.random.default_rng(23)
anchors_x = [5, 40, 70, 95, 125, 150, 178, 205, 232]
anchors_y = [0.10, 0.24, 0.31, 0.44, 0.52, 0.63, 0.71, 0.85, 0.965]
yfull = []
for i in range(len(anchors_x)-1):
    m = anchors_x[i+1]-anchors_x[i]
    base = np.linspace(anchors_y[i], anchors_y[i+1], m)
    seg = base + 0.030*np.convolve(rng2.standard_normal(m), np.ones(9)/9, mode="same") \
        + 0.008*rng2.standard_normal(m)
    yfull.append(seg)
yfull = np.clip(np.concatenate(yfull), 0.02, 1.02)
ax.plot(np.arange(anchors_x[0], anchors_x[-1]), yfull, color=VERM, lw=2.4, alpha=0.95, zorder=5,
        solid_capstyle="round")

# labels
ax.text(2, 0.045, "quiescent flow", fontsize=10.5, color=BLUE, alpha=0.9)
ax.text(2, 0.995, "hurricane genesis", fontsize=11, color=VERM, fontweight="bold")
for i, lam in enumerate(interfaces):
    ax.text(n-2, lam+0.015, rf"$\lambda_{i+1}$", fontsize=11, color="#6a7680", ha="right")
ax.text(n-2, target_lo+0.015, r"$\lambda_{\mathrm{gen}}$", fontsize=11, color=VERM, ha="right")

# cyclone glyph: solid eye + two open comma arms (classic TC symbol)
gx = fig.add_axes([0.545, 0.585, 0.050, 0.11])
gx.set_aspect("equal"); gx.axis("off")
gx.add_patch(plt.Circle((0, 0), 0.34, fill=True, color=VERM))
for sgn in (1, -1):
    t = np.linspace(0, 1, 60)
    r = 0.34 + 1.05*t
    th = (np.pi*0.15 if sgn > 0 else np.pi*1.15) + 1.35*t
    gx.plot(sgn*np.abs(r)*np.cos(th)*(1 if sgn>0 else 1), r*np.sin(th) if False else r*np.sin(th), color=VERM, lw=3.0, solid_capstyle="round") if False else None
    gx.plot(r*np.cos(th), r*np.sin(th), color=VERM, lw=3.0, solid_capstyle="round")
gx.set_xlim(-1.15, 1.15); gx.set_ylim(-1.15, 1.15)

ax.set_xlim(0, n); ax.set_ylim(0, 1.06)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("time", fontsize=11, color="#5a6a76")
ax.set_ylabel("storm intensity  $\\lambda$", fontsize=11, color="#5a6a76")
for s in ax.spines.values(): s.set_color("#c8cfd4")

# ---------------- right panel: heavy tail ----------------
ax2 = fig.add_axes([0.665, 0.13, 0.285, 0.60])
xv = np.linspace(-3.6, 6.0, 500)
pdf = np.exp(-0.5*xv**2)
smooth_on = 1.0/(1.0 + np.exp(-(xv-1.3)/0.45))
tail = 0.14*np.exp(-0.75*(xv-1.3))*smooth_on
f = pdf + tail
f /= f.max()
ax2.plot(xv, f, color=INK, lw=2.0)
cut = 2.3
ax2.fill_between(xv[xv >= cut], f[xv >= cut], color=VERM, alpha=0.75, lw=0)
ax2.annotate("the tails:\nrare, high-impact\nweather", xy=(3.15, 0.045), xytext=(2.1, 0.42),
             fontsize=11, color=INK, ha="left",
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
ax2.text(-3.3, 0.93, "forecast distribution", fontsize=10.5, color="#5a6a76")
ax2.set_xlim(-3.6, 6.0); ax2.set_ylim(0, 1.05)
ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values(): s.set_color("#c8cfd4")

out = "/glade/work/schreck/repos/miles-tails/images/tails_new"
fig.savefig(out + ".png", dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(out + ".svg", facecolor="white", bbox_inches="tight")
print("saved", out + ".png/.svg")
