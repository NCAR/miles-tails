# Assessment: Validity of FFS Applied to Hurricane Genesis
**Based on:** `main.tex` — *Forward Flux Sampling for Rare Atmospheric Events: Bridging Statistical Mechanics and Neural Weather Emulation*

---

## Overview

The paper's central theoretical claim is that Forward Flux Sampling (FFS) is **more** appropriate for atmospheric rare events than equilibrium-based methods, because FFS was designed for non-equilibrium steady state (NESS) systems and makes no assumptions about detailed balance, Boltzmann statistics, or free energy landscapes. This memo evaluates that claim against the original FFS methodology and the broader non-molecular applications literature.

---

## 1. What the Original FFS Papers Actually Say

### Was FFS designed for NESS systems?

**Yes — this claim is well-founded.**

The foundational Allen et al. 2005 paper (*Physical Review Letters* 94, 018104) introduced FFS explicitly to handle bistable **biochemical networks that are out of thermodynamic equilibrium**. The motivation was precisely that earlier path-sampling methods (Transition Path Sampling, Transition Interface Sampling) required microscopic reversibility and could not be applied to non-equilibrium systems. FFS was designed to escape this constraint by using only *forward* trajectories and measuring directional flux rather than equilibrium populations.

The Allen et al. 2006 paper (*J. Chem. Phys.* 124, 024102) is titled "Simulating rare events in **equilibrium or nonequilibrium** stochastic systems" — the dual applicability is explicit in the title and framing.

The 2009 review (*J. Phys.: Condens. Matter* 21, 463102) states directly: "FFS does not require backward pathways, hence eliminating the need for the backward shooting part of TPS or TIS; **this is the feature that allows FFS to be applicable to non-equilibrium systems**."

**Verdict: The paper's argument that FFS was designed for NESS is accurate and well-supported by the primary literature.**

---

### What are the actual requirements for FFS?

The paper states FFS requires only:
1. A well-defined stochastic or chaotic dynamical system
2. Distinguishable metastable regions in state space
3. Measurable flux between those regions
4. Markovian dynamics

This is largely correct, but the literature identifies a few additional practical requirements worth noting:

- **Smooth order parameter**: The order parameter should vary smoothly so that interface crossings can be detected cleanly. Min-MSLP is smooth at 6-hour resolution.
- **Timescale separation**: Residence time in each metastable region should be long compared to the typical transition time. The paper addresses this explicitly — storms reside in the disturbance state for days to weeks, while genesis takes 1–3 days.
- **Effective decorrelation**: Saved configurations at each interface must be statistically independent. The paper enforces this via a decorrelation flag requiring trajectories to return to state A before the next crossing is recorded.

The paper addresses all of these in Sections 2 and 4. No significant gaps here.

---

## 2. Non-Molecular Applications of FFS in the Literature

### A. Biochemical and gene regulatory networks (strong precedent)

FFS has been applied extensively to stochastic biochemical systems:

- **Bistable genetic toggle switches**: FFS is used to compute switching rates between gene expression states. These are explicitly non-equilibrium (driven by continuous transcription/degradation cycles). This is the direct motivation for the 2005 paper.
- **Transcription factor dimerization**: FFS samples rare binding/unbinding events in gene regulatory networks.
- **Master equation models**: Automatic error control methods have been developed for FFS on discrete stochastic biochemical systems.

These are genuine NESS applications — the biochemical systems are maintained far from equilibrium by continuous enzymatic reactions, exactly analogous (in spirit) to the atmosphere being maintained far from equilibrium by solar forcing.

### B. Atmospheric and climate science (important nuance)

Here the picture is more nuanced. **The atmospheric science rare-event community predominantly uses splitting algorithms rather than FFS specifically.** Key examples:

- **Adaptive Multilevel Splitting (AMS)**: Applied to extreme heatwaves, cold spells, blocking onset, and jet stream transitions. The Ragone et al. 2018 PNAS paper (cited in `main.tex`) uses a large deviation algorithm related to AMS, not FFS.
- **Quantile Diffusion Monte Carlo**: The Webber et al. 2019 *Chaos* paper (also cited in `main.tex`) applies this to hurricane intensity, not FFS.
- **TEAMS (trying-early AMS)**: Developed specifically for sudden, transient atmospheric extremes like stratospheric warmings.

**There is no prior published paper applying FFS specifically to hurricane genesis or to any tropical cyclone problem.** The paper is genuinely novel in this respect.

**However, this is not a fatal flaw.** AMS and FFS are algorithmically related — both are interface-based splitting methods that factorize a rare transition into a product of more probable steps. The key mathematical operations are the same. FFS and AMS differ mainly in:
- FFS uses fixed interfaces and counts flux; AMS uses adaptive interfaces and resamples
- FFS is designed for steady-state rate estimation; AMS is more flexible for transient events
- FFS has stronger theoretical guarantees for NESS steady-state flux

For a system like tropical cyclogenesis — which is about estimating a steady-state *rate* (how many genesis events per season) rather than a single transient episode — FFS is arguably **more appropriate** than AMS or quantile DMC, because the flux-based rate estimator is exactly what's needed. The paper makes this argument implicitly by framing genesis as a steady-state rate problem, and it's a good choice.

### C. Other non-molecular domains

- **Epidemiology**: No direct FFS applications found. Splitting-type methods are used for disease extinction/invasion rates but not FFS proper.
- **Population dynamics/ecology**: Weighted ensemble methods are used but not FFS.
- **Social/economic systems**: No FFS applications found.
- **Fluid dynamics/turbulence**: AMS has been used for turbulent transitions (pipe flow transition to turbulence), but not FFS.

---

## 3. The Key Theoretical Question: Stochastic vs. Chaotic Dynamics

This is the most important subtlety. The original FFS papers are framed around **stochastic** systems — dynamics include explicit noise terms (Langevin equations, master equations, or Brownian dynamics). The paper applies FFS to a **deterministic chaotic** system (neural weather emulator with stochastic layers), arguing that chaotic sensitivity to initial conditions provides "effective stochasticity."

**Is this justified?**

Yes, and here is why:

1. **The Allen et al. 2009 review explicitly states FFS applies to "deterministic chaotic systems where sensitive dependence on initial conditions provides effective stochasticity."** This is not an extrapolation by the paper's authors — it is directly stated in the method's own review paper.

2. **Ensemble spread from chaotic divergence is mathematically equivalent to ensemble spread from stochastic noise for FFS purposes.** Both produce a distribution of outcomes at each interface crossing. FFS only cares that this distribution is well-sampled and that saved configurations are decorrelated — it does not care whether spread comes from noise or chaos.

3. **The SDL-WXFormer adds explicit stochastic decomposition layers**, so the system is not purely deterministic — it has genuine stochastic noise in the ensemble generation. This makes the system cleanly analogous to the stochastic systems FFS was originally designed for.

4. **Lyapunov decorrelation**: The paper correctly argues that atmospheric Lyapunov exponents (~0.01/day) imply decorrelation timescales of 3–5 days, so saved interface configurations are effectively independent when trajectories return to state A. This satisfies the decorrelation requirement.

**One legitimate concern**: In purely deterministic chaotic systems without noise, the ensemble spread depends entirely on the initial perturbation distribution. If perturbations are too small, most trajectories look identical and FFS is inefficient. If too large, you lose accuracy near the interfaces. The paper addresses this by using the SDL-WXFormer's stochastic decomposition layers to generate perturbations with calibrated amplitude and structure — a reasonable approach.

---

## 4. The Non-Equilibrium Argument: Is It Well-Constructed?

The paper's core argument (Section 2.1, 2.5, and 5.3) is:

> The atmosphere is a NESS, not an equilibrium system. Equilibrium concepts (Boltzmann distributions, detailed balance, free energy) don't apply. FFS doesn't use these concepts. Therefore FFS is valid for the atmosphere, whereas equilibrium-based methods are not.

**Assessment: This argument is structurally sound and represents a genuine contribution to the literature.**

The contrast the paper draws between equilibrium and non-equilibrium rarity is important and underappreciated:

- In molecular systems: rarity is **energetic** — rare states have high free energy, so thermal fluctuations rarely access them. Rates follow Arrhenius scaling.
- In the atmosphere: rarity is **geometric/dynamical** — rare states occupy small volumes on the attractor, visited infrequently due to the structure of the governing dynamics, not because they have "high energy." Hurricanes actually release enormous stored potential energy during genesis.

This distinction matters because it rules out methods that rely on the Boltzmann distribution to define "rare" (e.g., umbrella sampling, metadynamics, Jarzynski-based methods). FFS, operating purely on fluxes and conditional probabilities, is immune to this concern.

**The large deviation theory connection** (Section 5.3) is also well-founded. Large deviation theory provides the rigorous framework for rare event probabilities in non-equilibrium stochastic systems, and FFS can be interpreted as importance sampling within the large deviation framework. This is consistent with the broader rare event methods literature.

---

## 5. Strengths and Gaps in the Paper's Theoretical Argument

### Strengths

| Claim | Assessment |
|---|---|
| FFS was designed for NESS systems | ✅ Correct — verified in Allen et al. 2005, 2006, 2009 |
| FFS requires no equilibrium assumptions | ✅ Correct — this is an explicit design feature |
| Atmospheric chaos provides effective stochasticity | ✅ Supported by Allen et al. 2009 directly |
| Rarity in atmosphere is dynamical, not energetic | ✅ Physically correct and important distinction |
| Strange attractor framework for metastability | ✅ Theoretically coherent and appropriate |
| Timescale separation exists for genesis | ✅ Empirically supported (days vs. weeks) |
| Decorrelation from Lyapunov divergence | ✅ Quantitatively justified with ~3–5 day timescale |

### Gaps Worth Addressing

1. **Why FFS rather than AMS/quantile DMC?** The atmospheric rare event community has converged on AMS and splitting algorithms. The paper should explicitly state why FFS is preferred: (a) steady-state *rate* estimation is the goal (not a single extreme episode), and (b) the factorization $k_{AB} = \Phi_{A,0} \prod P(\lambda_{i+1}|\lambda_i)$ gives direct physical interpretation of each barrier. This is a defensible and important point that the paper leaves implicit.

2. **No direct citation of FFS applied to deterministic chaos**. While Allen et al. 2009 does mention deterministic chaotic systems, the paper would be strengthened by citing specific studies that successfully applied interface methods to chaotic dynamical systems (e.g., pipe flow turbulence, Lorenz-96 models).

3. **The Markovian assumption deserves more scrutiny**. The paper lists Markovian dynamics as a requirement but does not test it. Atmospheric systems have memory (SST anomalies, soil moisture, stratospheric state) that can violate the Markov property at longer timescales. For 10-day forecasts this is likely not a serious issue, but an acknowledgment would strengthen the paper.

4. **Order parameter optimality**: The paper acknowledges MSLP is not the optimal reaction coordinate and points to future work on learned committors. This is appropriately modest. The 85% monotonicity statistic is a good empirical check.

5. **The "biochemical network" analogy could be drawn more explicitly**. The 2005 Allen et al. paper's bistable genetic switch is a perfect analog for tropical cyclogenesis: a driven non-equilibrium system with two distinct states (active/inactive gene expression; organized/unorganized vortex) separated by a barrier that is dynamical rather than energetic. Making this analogy explicit would strengthen the reader's intuition.

---

## 6. Overall Verdict

**The argument for using FFS for hurricane genesis is valid and well-grounded in the original FFS methodology.**

The paper correctly identifies the key features of FFS that make it applicable beyond molecular systems — it was explicitly designed for NESS dynamics, requires no equilibrium assumptions, and its authors stated it applies to deterministic chaotic systems. The application to hurricane genesis is novel (no prior published precedent exists in exactly this form), but it is a natural extension of the method's stated scope.

The primary theoretical gap is not in the validity of FFS per se, but in the paper's failure to explicitly compare FFS to the splitting algorithms (AMS, quantile DMC) that the atmospheric science community already uses. This comparison would:
- Position the work relative to Ragone et al. 2018 and Webber et al. 2019 (both cited but not compared algorithmically)
- Justify the choice of FFS over AMS for steady-state rate estimation
- Acknowledge the existing atmospheric rare event sampling literature more completely

**The core physics and statistics are sound.** The atmosphere is a NESS, FFS is designed for NESS, the attractor-based metastability framework is appropriate, and the implementation details (decorrelation, CPS filtering, multi-storm tracking) are carefully designed. The empirical results (agreement with IFS at λ₀, physically reasonable conditional probabilities of 0.34–0.62, correlation with wind shear) provide strong validation.

---

## Key References for Context

**Original FFS Papers:**
- Allen, R.J., Warren, P.B., ten Wolde, P.R. (2005). *Sampling rare switching events in biochemical networks.* PRL 94, 018104. ← Explicitly NESS biochemical systems
- Allen, R.J., Frenkel, D., ten Wolde, P.R. (2006). *Simulating rare events in equilibrium or nonequilibrium stochastic systems.* JCP 124, 024102.
- Allen, R.J., Valeriani, C., ten Wolde, P.R. (2009). *Forward flux sampling for rare event simulations.* J. Phys.: Condens. Matter 21, 463102. ← Mentions deterministic chaotic systems explicitly

**Atmospheric Rare Event Methods (the community context):**
- Ragone, F., Wouters, J., Bouchet, F. (2018). *Computation of extreme heat waves in climate models using a large deviation algorithm.* PNAS 115, 24–29. ← AMS-type, not FFS
- Webber, R.J. et al. (2019). *Practical rare event sampling for extreme mesoscale weather.* Chaos 29, 053109. ← Quantile DMC, not FFS
- Finkel, J. et al. (2024). *Bringing statistics to storylines: rare event sampling for sudden, transient extreme events.* JAMES. ← TEAMS/AMS for transient events

**Theoretical Foundation:**
- Touchette, H. (2009). *The large deviation approach to statistical mechanics.* Physics Reports 478, 1–69.
- Lorenz, E.N. (1969). *Atmospheric predictability as revealed by naturally occurring analogues.* JAS 26, 636–646. ← The Lyapunov/chaos foundation cited in the paper

---

*Memo prepared: March 2026*
