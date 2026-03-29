---
title: FYS4411 - Project 1
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    template: ../../report_template
    output: report.pdf
    showtoc: true
math:
  # Note the 'single quotes'
  '\argmin': '\operatorname{argmin}'
  '\drm': '\mathrm{d}'
  '\N': '\mathbb{N}'
  '\R': '\mathbb{R}'
  '\unitvec': '{\hat{\mathbf{#1}}}'
bibliography: references.bib
abstract: |
  We apply variational Monte Carlo (VMC) and steepest descent optimization to estimate the ground‑state energy of a Bose gas with Jastrow-type repulsive interactions. The trial wavefunction is sampled using Metropolis importance sampling driven by Langevin dynamics, which incorporates drift terms to guide walkers toward regions of high probability density. Our optimization results captures the expected spatial expansion of the condensate induced by repulsive interactions.
---

# Introduction

In this report, we investigate the use of variational Monte Caro to compute the ground‑state energy of a Bose gas with Jastrow‑type repulsive interactions. A key component of any Monte Carlo simulation is the sampling strategy used to explore the probability distribution defined by the trial wavefunction. We therefore employ and compare two different sampling techniques: the brute‑force Metropolis algorithm with symmetric proposal moves, and an importance‑sampling scheme driven by Langevin dynamics. While the brute‑force approach relies on unbiased random displacements, importance sampling incorporates drift terms that guide walkers toward regions of high probability density, potentially improving sampling efficiency and reducing statistical fluctuations.

To validate and calibrate these methods, we first apply them to the non‑interacting harmonic oscillator, a system with an analytically known ground state. This benchmark allows us to assess the accuracy, stability, and convergence properties of each sampling strategy. Having established a reliable computational framework, we then extend the analysis to the interacting Bose gas. By combining VMC with steepest‑descent optimization, we determine the optimal Gaussian confinement parameter in the trial wavefunction and examine how it responds to repulsive correlations.

# Theory and Method

## Bose Gas Correlated Model

We consider a system of $N$ identical bosons of mass $m$, confined in a harmonic trapping potential an interacting in a hard-sphere pair potential. The system is represented by the bosonic Hilbert space $\mathcal{H}_N = L_\text{sym}^2 (\R^{3N})$, which is the subspace of $L^2 (\R^{3N})$ consisting of totally symmetric wave functions. 

Let $\mathbf{r}_i \in (x_i, y_i, z_i) \in\R^3$ be the position of the $i$th particle. The $N$-body Hamiltonian $\hat{H}$ is given by

$$
\label{equation:hamiltonian}
  \hat{H} = \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \nabla_i^2 + V_\text{ext}(\mathbf{r}_i) \right) + \sum_{i=1}^N \sum_{j=i+1}^N V_\text{int} (\mathbf{r}_i, \mathbf{r}_j)
$$

The external potential $V_\text{ext} : \R^3 \to\R$ is taken to be either a spherical (S) or elliptical (E) trap: 

$$
\label{equation:trap-potential}
  V_\text{ext} (\mathbf{r}) = \begin{cases}
    \frac{1}{2} m\omega_\text{ho}^2 r^2,\quad& (S) \\
    \frac{1}{2} m[\omega_\text{ho}^2 (x^2 + y^2) + \omega_z^2 z^2],\quad& (E)
  \end{cases},
$$

where $r = |\mathbf{r}|$, $\omega_\text{ho} > 0$ is the radial trap frequency in the $xy$-plane, and $\omega_z > 0$ is the axial frequency in the $z$-direction. Note that the spherical case corresponds to $\omega_z = \omega_\text{ho}$.

The inter-boson interaction is given by a pairwise, repulsive potential

$$
  V_\text{int} (|\mathbf{r}_i - \mathbf{r}_j|) = \begin{cases}
    \infty,\quad& |\mathbf{r}_i - \mathbf{r}_j| \leq a \\
    0,\quad& |\mathbf{r}_i - \mathbf{r}_j| > a
  \end{cases},
$$

where $a \geq 0$ is the hardcore diameter of the bosons. The interaction $V_\text{int}$ is zero if the bosons are separated by a distance $|\mathbf{r}_i - \mathbf{r}_j|$ greater than $a$, but infinite if they attempt to come within a distance $|\mathbf{r}_i - \mathbf{r}_j| \leq a$.

To approximate the ground state using variational Monte Carlo method, we employ a Jastrow-type trial wave function $\Psi_T$ of the form

$$
\label{equation:trial-wavefunction}
  \Psi_T (\mathbf{r}_1,\dots,\mathbf{r}_N,\alpha,\beta) = \left(\prod_{i=1}^N g(\alpha, \beta, \mathbf{r}_i) \right) \left(\prod_{j < k} f(a, |\mathbf{r}_j - \mathbf{r}_k|) \right),
$$

where $\alpha,\beta\in\R$ are variational parameters. The single-particle wave function is proportional to the harmonic oscillator for the ground state, i.e.

$$
  g(\alpha,\beta, \mathbf{r}_i) = \exp[-\alpha(x_i^2 + y_i^2 + \beta z_i^2)]
$$

For spherical traps, we have $\beta = 1$ and for non-interacting bosons ($a = 0$) we have $\alpha = 1/2a_\text{ho}^2$. The correlation wave function is

$$
\label{equation:correlation-wave-function}
  f(a,|\mathbf{r}_i - \mathbf{r}_j|) = \begin{cases}
    0,\quad& |\mathbf{r}_i - \mathbf{r}_j| \leq a \\
    1 - \frac{a}{|\mathbf{r}_i - \mathbf{r}_j|},\quad& |\mathbf{r}_i - \mathbf{r}_j| > a
  \end{cases}
$$

### Dimensionless Hamiltonian

For numerical analysis, it is convenient to express the Hamiltonian [](#equation:hamiltonian) in dimensionless form. This can be done by introducing dimensionless coordinates $\mathbf{r}' = \mathbf{r}/a_\text{ho}$, where

$$
  a_\text{ho} := \sqrt{\frac{\hbar}{m\omega_\text{ho}}}
$$

is the characteristic length of a spherical trap. Substituting $\mathbf{r} = a_\text{ho} \mathbf{r}$ into the external potential [](#equation:trap-potential), the ellictical trap case becomes

$$
\begin{align*}
  V_\text{ext} (\mathbf{r}) =& \frac{1}{2} \underbrace{m a_\text{ho}^2}_{\hbar/\omega_\text{ho}} [\omega_\text{ho} ({x'}^2 + {y'}^2) + \omega_z^2 {z'^2}] \\
  =& \frac{1}{2} \hbar \omega_\text{ho} \left[{x'}^2 + {y'}^2 + \left(\frac{\omega_z}{\omega_\text{ho}} \right)^2 {z'}^2 \right]
\end{align*}
$$

In terms of the dimensionless coordinates, the gradient and Laplacian transform as

$$
  \nabla_i = \frac{1}{a_\text{ho}} \nabla'_i,\quad \nabla_i^2 = \frac{1}{a_\text{ho}^2} {\nabla'_i}^2
$$

The single-particle kinetic term therefore becomes

$$
  -\frac{\hbar^2}{2m} \nabla_i^2 = -\frac{\hbar^2}{2m}\frac{1}{a_\text{ho}} {\nabla'_i}^2 = -\frac{\hbar}{2m}{m\omega_\text{ho}}{\hbar} {\nabla'_i}^2 = -\frac{\hbar\omega_\text{ho}}{2} {\nabla'_i}^2
$$

Dividing by the charateristic energy scale $\hbar\omega_\text{ho}$ and introducing the anisotropoy parameter $\gamma = \omega_z / \omega_\text{ho}$, we obtain the dimensionless many-body Hamiltonian

$$
\label{equation:hamiltonian-dimensionless}
  \hat{H} = \sum_{i=1}^N \frac{1}{2} (-\nabla_i^2 + x_i^2 + y_i^2 + \gamma^2 z_i^2) + \sum_{i < j} V_\text{int} (|\mathbf{r}_i - \mathbf{r}_j|)
$$

The special case $\gamma = 1$ corresponds to an spherical (isotropic) trap.

### Local Energy

To prepare the variational Monte Carlo computation, we derive the local energy energy associated with the trial wave function. Introducing $\mathbf{R} = (\mathbf{r}_1,\dots,\mathbf{r}_N) \in\R^{3N}$, the local energy is defined pointwise by

$$
\label{equation:local-energy}
  E_L (\mathbf{R}) = \frac{1}{\Psi_T (\mathbf{R})} \hat{H} \Psi_T (\mathbf{R}),\; \Psi_T (\mathbf{R}) \neq 0.
$$

#### Non-interacting Case

In the non-interacting case $a = 0$, the inter-boson interaction $V_\text{int}$ vanishes and the correlation wave function reduces to unity, $f(|\mathbf{r}_i - \mathbf{r}_j|) = 1$. The trial wavefunction [](#equation:trial-wavefunction) therefore becomes a simple product of identical one-body orbitals,

$$
\label{equation:trial-wavefunction-noninteracting}
  \Psi_T (\mathbf{R}) = \prod_{i=1}^N g(\alpha, \beta, \mathbf{r}_i).
$$

In this case the Hamiltonian [](#equation:hamiltonian) reduces to a sum of independent one-body operators

$$
  \hat{H}_0 = \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \nabla_i^2 + V_\text{ext}(\mathbf{r}_i) \right),
$$

so that the system consists of $N$ non-interacting bosons in the external trapping potential.

For a spherical trap potential corresponding to $\beta = 1$, we can derive the exact solution for the ground state energy. In this case, the trial wave function [](#equation:trial-wavefunction-noninteracting) factorizes as

$$
\label{equation:trial-wavefunction-noninteracting-isotropic}
  \Psi_T (\mathbf{R}) = \prod_{i=1}^N e^{-\alpha r_i^2} = \exp\left(\alpha \sum_{i=1}^N r_i^2 \right),\; r_i = |\mathbf{r}_i| =.
$$

Defining the one-particle factor $\phi(\mathbf{r}_i) := \exp(-\alpha r_i^2)$, the Laplacian of $\Psi_T$ can be shown to satisfy

$$
  \frac{\nabla_i^2 \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)}
$$

Applying this identity, the local energy [](#equation:local-energy) simplifies to

$$
\label{equation:local-energy-noninteracting}
  E_L (\mathbf{R}) = \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)} + V_\text{ext} (\mathbf{r}_i) \right),
$$

which separates into a sum of identical single-particle contributions. Evaluating the Laplacian $\nabla_i \phi(\mathbf{r}_i)$ for each $i$ yields

$$
  \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)} = 4\alpha^2 r_i^2 - 2d\alpha,
$$

where $d$ is the spatial dimension of the system.

Inserting the spherical trap potential from [](#equation:trap-potential), the local energy [](#equation:local-energy-noninteracting) becomes

$$
\label{equation:local-energy-noninteracting-analytic}
  E_L (\mathbf{R}) = \frac{d\hbar^2 \alpha N}{m} + \sum_{i=1}^N \left(\frac{1}{2} m\omega_\text{ho}^2 - \frac{2\hbar^2 \alpha^2}{m} \right) r_i^2.
$$

The sum vanishes when

$$
  \alpha = \frac{m\omega_\text{ho}}{2\hbar},
$$

which corresponds to the exact ground-state width of the harmonic-oscillator. In this case, the local energy becomes a constant,

$$
\label{equation:ground-state-energy-noninteracting}
  E_L = \frac{d}{2} N\hbar \omega_\text{ho},
$$

which coincides with the exact ground state-energy of $N$ non-interacting bosons in a spherical harmonic trap.

In addition the analytic expression [](#equation:local-energy-noninteracting-analytic), the local energy can be numerically approximated using finite difference derivation to compute the Laplacian $\nabla^2 \phi$ in [](#equation:local-energy-noninteracting). The second-order central difference approximation is given by

$$
\label{equation:second-order-finite-difference}
  \nabla^2 \phi (\mathbf{r}) \approx \sum_{j=1}^d \frac{\phi(\mathbf{r} + h\mathbf{e}_j) - 2\phi(\mathbf{r}_j) + \phi(\mathbf{r}_i - h\mathbf{e}_j)}{h^2}
$$

where $h$ is the finite-difference step size, and $\mathbf{e}_j$ is the unit vector in the $j$th coordinate direction.

#### Interacting Case

In the interacting case $a > 0$, the local energy takes the form

$$
  E_L (\mathbf{R}) = \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \frac{\nabla_i^2 \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} + V_\text{ext}(\mathbf{r}_i) \right) + \sum_{i=1}^N \sum_{j=i+1}^N V_\text{int} (\mathbf{r}_i, \mathbf{r}_j)
$$

Applying the dimensionless Hamiltonian [](#equation:hamiltonian-dimensionless), this can be recast as

$$
  E_L (\mathbf{R}) = \sum_{i=1}^N \frac{1}{2} \left(-\frac{\nabla^2 \Psi_T(\mathbf{R})}{\Psi_T (\mathbf{R})} + x_i^2 + y_i^2 + \gamma^2 z_i^2 \right) + \sum_{i < j} V_\text{int} (|\mathbf{r}_i - \mathbf{r}_j|)
$$

As shown in [](#Logarithmic Gradient of the Trial Wavefunction), the analytic expression for the logarithmic Laplacian of $\Psi_T$ is

$$
\label{equation:trial-wavefunction-interacting-laplacian}
\begin{split}
  \frac{1}{\Psi_T (\mathbf{R})} \nabla_i^2 \Psi_T (\mathbf{R}) =& -2\alpha(2 + \beta) + 4\alpha^2 (x_i^2 + y_i^2 + \beta^2 z_i^2) \\
  &- 4\alpha a \sum_{j\neq i} \frac{(x_i - x_j)x_k + (y_i - y_j)y_k + \beta(z_i - z_j)z_k}{r_{ij}^2 (r_{ij} - a)} \\
  &+ a^2 \sum_{j\neq i} \sum_{k\neq i} \frac{(\mathbf{r}_i - \mathbf{r}_j)\cdot(\mathbf{r}_i - \mathbf{r}_k)}{r_{ij}^2 r_{ik}^2 (r_{ij} - a)(r_{ik} - a)} \\
  &- a^2 \sum_{j\neq i} \frac{1}{r_{ij}^2 (r_{ij} - a)^2}
\end{split}
$$

### One-body Density

For a many-body system of $N$ particles represented by a normalized wavefunction $\Psi_T$, the one-body reduced density associated with particle $i$ is defined as the marginal of the probability measure $|\Psi_T|^2 \drm\mathbf{r}_1 \cdots \drm\mathbf{r}_N$ onto the $i$th coordinate

$$
  \rho_i (\mathbf{r}) = N \int_{\R^{3(N-1)}} |\Psi(\mathbf{r}_1,\dots,\mathbf{r}_{i-1},\mathbf{r},\mathbf{r}_{i+1},\dots,\mathbf{r}_N)|^2 \; \prod_{j\neq i} \drm\mathbf{r}_j
$$

Since bosonic states are symmetric, the $\rho_i$ are independent of the particle label $i$, and we simply write $\rho(\mathbf{r})$. In a spherical trap with $\beta = 1$ without interactions, the one-body density takes the form

$$
  \rho(\mathbf{r}) = N \left(\frac{2\alpha}{\pi} \right)^{3/2} e^{-2\alpha r^2}\; r = |\mathbf{r}|.
$$

In particular, the harmonic oscillator ground state corresponds to $\alpha = 1/2$, resulting in the one-body density

$$
  \rho(r) = \frac{N}{\pi^{3/2}} e^{-r^2}
$$

## Variational Monte Carlo Estimation

The ground state energy of the bose gas correlated model can be estimated using variational Monte Carlo methods. For a trial wavefunction $\Psi_T (\mathbf{R}; \boldsymbol{\alpha})$, the variational energy is defined as the expectation value of the Hamiltonian $\hat{H}$:

$$
\label{equation:hamiltonian-expectation}
\begin{split}
  E(\boldsymbol{\alpha}) :=& \braket{H}_{\Psi_T} = \frac{\braket{\Psi_T (\mathbf{R}; \boldsymbol{\alpha}), \hat{H} \Psi_T (\mathbf{R}, \boldsymbol{\alpha})}}{\braket{\Psi(\mathbf{R}, \boldsymbol{\alpha}), \Psi_T (\mathbf{R}; \boldsymbol{\alpha})}} \\
  =& \frac{\int_{\R^{3N}} \Psi_T^* (\mathbf{R}; \boldsymbol{\alpha}) \hat{H}\Psi_T (\mathbf{R}; \boldsymbol{\alpha}) \;\drm\mathbf{R}}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}; \boldsymbol{\alpha})|^2 \;\drm\mathbf{R}},
\end{split}
$$

where $\boldsymbol{\alpha}$ denotes a vector of variational parameters. By the Rayleigh-Ritz principle $E(\boldsymbol{\alpha}) \geq E_0$, where $E_0$ is the ground-state energy. Thus, we can approximate $E_0$ by minimizing $E(\boldsymbol{\alpha})$ over the parameters $\boldsymbol{\alpha}$.

To enable Monte Carlo estimation, we expand the variational energy [](#equation:trial-wavefunction) in terms of the probability density function

$$
\label{equation:wavefunction-pdf}
  P_{\boldsymbol{\alpha}} (\mathbf{R}) = \frac{|\Psi_T (\mathbf{R}; \boldsymbol{\alpha})|^2}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}; \boldsymbol{\alpha})|^2 \;\drm\mathbf{R}}
$$

Substituting the local energy [](#equation:trial-wavefunction), the variational energy can be written as the expectation value

$$
\label{equation:variation-energy-expectation}
  E(\boldsymbol{\alpha}) = \int_{\R^{3N}} P_{\boldsymbol{\alpha}} (\mathbf{R}) E_L (\mathbf{R}; \boldsymbol{\alpha}) \;\drm\mathbf{R} = \mathbb{E}_{P_{\boldsymbol{\alpha}}} (E_L)
$$

Consequently, computing $E(\boldsymbol{\alpha})$ reduces to sampling from $P_{\boldsymbol{\alpha}} (\mathbf{R}) \propto |\Psi_T (\mathbf{R}, \boldsymbol{\alpha})|^2$. However, the normalization constant

$$
  Z = \int_{\R^{3N}} |\Psi_T (\mathbf{R}, \boldsymbol{\alpha})|^2 \;\drm\mathbf{R}
$$

is intractable in high dimensions, making direct sampling from $P_{\boldsymbol{\alpha}}$ unfeasible. To overcome this, we can employ the Metropolis-Hastings algorithm to construct a Markov chain $\set{\mathbf{R}^{(k)}}_{k\geq 0}$ whose stationary distribution is $P_{\boldsymbol{\alpha}}$.

Given the current state $\mathbf{R}$, we propose a move $\mathbf{R}' \sim T(\cdot|\mathbf{R})$, where $T$ is a chosen transition kernel. The proposal is accepted with probability

$$
  A(\mathbf{R},\mathbf{R}') = \min\Set{1, \frac{\lvert\Psi_T (\mathbf{R}', \boldsymbol{\alpha})\rvert^2 T(\mathbf{R} \lvert \mathbf{R}')}{\lvert\Psi_T (\mathbf{R}, \boldsymbol{\alpha})\rvert^2 T(\mathbf{R}' \lvert \mathbf{R})}}
$$

If the transition kernel is symmetric, i.e. $T(\mathbf{R}' | \mathbf{R}) = T(\mathbf{R}|\mathbf{R}')$, the acceptance probability simplifies to

$$
  A(\mathbf{R},\mathbf{R}') = \min\Set{1, \left\lvert\frac{\Psi_T (\mathbf{R}', \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}, \boldsymbol{\alpha})}\rvert^2},
$$

This special case corresponds to the original algorithm by [](article_metropolis_etal_1953), referred to as the *Metropolis algorithm*.

Using this procedure to generate samples $\set{\mathbf{R}_k}_{k=1}^M \sim P_{\boldsymbol{\alpha}}$, the expectation [](#equation:trial-wavefunction) approximates to the sample mean by the law of large numbers:

$$
\label{equation:energy-estimation}
  E(\boldsymbol{\alpha}) \approx \frac{1}{M} \sum_{k=1}^M E_L (\mathbf{R}^{(k)}; \boldsymbol{\alpha}) := \bar{E}_{\boldsymbol{\alpha}}, 
$$

where $M$ is the number of Monte Carlo samples. The Metropolis algorithm using a symmetric transition kernel can be summarized as follows:

Given parameters $\boldsymbol{\alpha}$:
1. Initialize the configuration $\mathbf{R}^{(0)}$
2. For $k = 0,\dots,M-1$:
    - Propose a new configurations $\mathbf{R}' \sim T(\cdot|\mathbf{R}^{(k)})$
    - Compute the acceptance ratio
$$
  A = \left|\frac{\Psi_T (\mathbf{R}', \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}^{(k)}, \boldsymbol{\alpha})}\right|^2
$$

    - Accept or reject:

$$
  \mathbf{R}^{(k + 1)} = \begin{cases}
    \mathbf{R}',\quad& \text{with probability } A, \\
    \mathbf{R}^{(k)},\quad& \text{with probability } 1 - A
  \end{cases}
$$

3. Estimate the energy using [](#equation:trial-wavefunction)

## Importance Sampling

To improve the efficiency of the random-walk Metropolis algorithm, we can apply importance sampling using Langevin molecular dynamics. A diffusion process characterized by a time-dependent probability density $P(\mathbf{R}, t)$ on $\R^{3N}$, is given by the Fokker-Planck equation

$$
\label{equation:fokker-planck}
  \frac{\partial P}{\partial t} = D \nabla \cdot (\nabla - \mathbf{F}) P(\mathbf{R}, t)
$$

where $\mathbf{F}$ is a drift term and $D > 0$ is the diffusion coefficient. The associated Langevin stochastic equation is

$$
  \frac{\partial \mathbf{R}(t)}{\partial t} = D \mathbf{F}(\mathbf{R}(t)) + \boldsymbol{\eta}(t)
$$

where $\boldsymbol{\eta}$ a random vector driven by a Wiener process. In discretized time steps $\Delta t$, the updated position is given by

$$
\label{equation:langevin-discretized}
  \mathbf{R}' = \mathbf{R} + D\mathbf{F}(\mathbf{R}) \Delta T + \xi \sqrt{\Delta t}
$$

where $\xi \sim \mathcal{N}(0,1)$ is a Gaussian random variable. In atomic units, the diffusion coefficient is set to $D = 1/2$ to reflect the kinetic energy term $-\frac{1}{2} \nabla^2$.

The probability density $P(\mathbf{R}, t)$ is stationary if and only if $\partial P/\partial t = 0$, implying that

$$
  \nabla \cdot (\nabla - \mathbf{F}) P(\mathbf{R}, t) = 0
$$

This is satisfied if the probability current vanishes, i.e. $\nabla P - \mathbf{F}P = 0$, resulting in the drift term

$$
  \mathbf{F}(\mathbf{R}) = \frac{\nabla P(\mathbf{R})}{P(\mathbf{R})} = \nabla \ln[P(\mathbf{R})]
$$

With a probabiliy density of the form $P(\mathbf{R}) \propto |\Psi_T (\mathbf{R})|^2$, we obtain

$$
\label{equation:quantum-force}
  \mathbf{F}(\mathbf{R}) = \nabla \ln[|\Psi_T (\mathbf{R})|^2] = 2 \frac{\nabla \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})}
$$

This quantity is known as the quantum force, which is introduces a drift towards regions of the configuration space where $|\Psi_T|^2$ is large. This significantly improves sampling efficiency compared with the Metropolis algorithm, which explores the configuration space through a symmetric random walk. 

The Fokker-Planck equation yields a transition kernel given by the Green's function

$$
  G(\mathbf{R}', \mathbf{R}, \Delta t) = \frac{1}{(4\pi D\Delta t)^{3N/2}} \exp\left(-\frac{[\mathbf{R}' - \mathbf{R} - D\Delta t \mathbf{F}(\mathbf{R})]^2}{4D\Delta t} \right)
$$

The acceptance probability for the Metropolis-Hastings algorithm now becomes

$$
  A(\mathbf{R}, \mathbf{R}') = \min\Set{1, \frac{G(\mathbf{R}, \mathbf{R}', \Delta t) \lvert\Psi_T (\mathbf{R}')\rvert^2}{G(\mathbf{R}', \mathbf{R}, \Delta t) \lvert\Psi_T (\mathbf{R})\rvert^2}}
$$

Introducing the shorthand

$$
\begin{equation*}
  \Delta\mathbf{R} = \mathbf{R}' - \mathbf{R},\quad \mathbf{F} = \mathbf{F}(\mathbf{R}), \quad \mathbf{F}' = \mathbf{F}(\mathbf{R}'),
\end{equation*}
$$

the ratio of Green's functions is given by

$$
\begin{align*}
  \frac{G(\mathbf{R}, \mathbf{R}', \Delta t)}{G(\mathbf{R}', \mathbf{R}, \Delta t)} =& \exp\left[-\frac{1}{4D\Delta t} \left( [-\Delta\mathbf{R} - D\Delta t \mathbf{F}']^2 - [\Delta\mathbf{R} - D\Delta t \mathbf{F}]^2 \right) \right] \\
  =& \exp\left[-\frac{1}{2} \Delta \mathbf{R} \cdot (\mathbf{F}' + \mathbf{F}) - \frac{D\Delta t}{4} \left(|\mathbf{F}'|^2 - |\mathbf{F}|^2 \right) \right]
\end{align*}
$$

Applying the identity $|\mathbf{F}'|^2 - |\mathbf{F}|^2 = (\mathbf{F}' + \mathbf{F}) \cdot (\mathbf{F}' - \mathbf{F})$, this can be refactored as

$$
  \frac{G(\mathbf{R}, \mathbf{R}', \Delta t)}{G(\mathbf{R}', \mathbf{R}, \Delta t)} = \exp\left[-\frac{1}{2} (\mathbf{F}' + \mathbf{F}) \cdot \left(\Delta\mathbf{R} + \frac{D\Delta t}{2} (\mathbf{F}' - \mathbf{F}) \right) \right],
$$

### Quantum Force

From [](#equation:quantum-force) the quantum force acting on particle $i$ is given by the logarithmic gradient of the trial wavefunction

$$
  \mathbf{F}_i = 2 \nabla_i \ln[\Psi_T (\mathbf{R})] = \frac{2\nabla_i \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})}
$$

#### Non-interacting case

For non-interacting system in a spherical trap, the trial wavefunction factorizes as 

$$
  \Psi_T (\mathbf{R}) = \prod_{i=1}^N \phi(\mathbf{r}_i),\; \phi(\mathbf{r}) = e^{-\alpha r^2}
$$

Taking the gradient with respect to particle $i$ yields

$$
  \nabla_i \Psi_T (\mathbf{R}) = \left(\prod_{j\neq i} \phi(\mathbf{r}_j) \right) \nabla_i \phi(\mathbf{r}_i)
$$

Dividing by $\Psi_T$, we obtain the identity

$$
  \frac{\nabla_i \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_i \phi (\mathbf{r}_i)}{\phi (\mathbf{r}_i)}
$$

Using [](#equation:gaussian-factor-spherical-log-gradient), the quantum force therefore becomes

$$
  \mathbf{F}_i = -4\alpha \mathbf{r}_i,
$$

which is purely radial and directed toward the origin.

#### Interacting Case

Using the analytic expression for the logarithmic gradient of $\Psi_T$ derived in [](#Logarithmic Gradient of the Trial Wavefunction), the quantum force in the interacting system becomes

$$
  \mathbf{F}_i = -4\alpha(x_i \unitvec{x} + y_i \unitvec{y} + \beta z_i \unitvec{z}) + 2a \sum_{j\neq k} \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}^2 (r_{kj} - a)}.
$$

The first term, corresponding to the single-particle elliptic confinement, is directed toward the origin as in the non-interacting case. In the isotropic limit $\beta = 1$, it reduces to the purely radial form $-4\alpha\mathbf{r}_i$. The second term, arising from the Jastrow factor, introduces repulsive particle interactions.

## Parameter Optimization

The optimal variational parameters $\bar{\boldsymbol{\alpha}}$ can be obtained by minimizing the variational energy $E(\boldsymbol{\alpha})$. This can be achieved by using the gradient $\nabla_{\boldsymbol{\alpha}} E(\boldsymbol{\alpha})$ with respect to variational parameters as the objective for an optimization procedure. As shown in [](#Energy Derivative Formula), the energy derivative with respect to a parameter $\alpha_k$ can be written as

$$
\label{equation:energy-derivatives}
  \frac{\partial E}{\partial\alpha_k} = 2 \left( \Braket{\frac{\partial_{\alpha_k} \Psi_T (\mathbf{R}; \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}; \boldsymbol{\alpha})} E_L (\mathbf{R}; \boldsymbol{\alpha})} - \Braket{\frac{\partial_{\alpha_k} \Psi_T (\mathbf{R}; \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}; \boldsymbol{\alpha})}} \braket{E_L (\mathbf{R}; \boldsymbol{\alpha})} \right)
$$

### Energy Derivatives

#### Non-interacting Case

In the non-interacting case with a spherical trap, the trial wavefunction depends only on the Gaussian confinement parameter $\alpha$. The logarithmic derivative with respect to $\alpha$ is therfore

$$
  \frac{\partial_{\alpha} \Psi_T (\mathbf{R}; \alpha)}{\Psi_T (\boldsymbol{R}; \alpha)} = \frac{\partial}{\partial\alpha} \ln \left[\exp\left(-\alpha \sum_{i=1}^N r_i^2 \right) \right] = -\sum_{i=1}^N r_i^2.
$$

#### Interacting Case

In the interacting case, the trial wavefunction also depends on the anisotrophy parameter $\beta$, in addition to $\alpha$. The logarithmic derivative with respect to $\alpha$ is

$$
\begin{align*}
  \frac{\partial_{\alpha} \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\boldsymbol{R}; \alpha)} =& \frac{\partial}{\partial\alpha} \ln\left[\exp\left(-\alpha \sum_{i=1}^N \left(x_i^2 + y_i^2 + \beta z_i^2 \right) \right) \prod_{j < k} f(a, r_{jk}) \right] \\
  =& \frac{\partial}{\partial\alpha} \left[-\alpha \sum_{i=1}^N \left(x_i^2 + y_i^2 + \beta z_i^2 \right) + \sum_{j < k} \ln[f(a, r_{jk})] \right].
\end{align*}
$$

Since the Jastrow correlation functions $f(a, r_{jk})$ are independent of $\alpha$, the second term vanishes under $\partial/\partial_\alpha$, resulting in 

$$
  \frac{\partial_{\alpha} \Psi_T (\mathbf{R}; \alpha)}{\Psi_T (\boldsymbol{R}; \alpha)} = -\sum_{i=1}^N \left(x_i^2 + y_i^2 + \beta z_i^2 \right).
$$

Similarly, the logarithmic derivative with respect to $\beta$ is

$$
  \frac{\partial_{\beta} \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\boldsymbol{R}; \alpha)} = -\alpha \sum_{i=1}^N z_i^2.
$$

# Results and Discussion

## Harmonic Oscillator

This section presents simulation results obtained using variational Monte Carlo to estimate the ground‑state energy of the harmonic oscillator, first using a brute‑force Metropolis sampler with a symmetric proposal and subsequently employing importance sampling.

### Parameter Grid Seach using Brute-force Metropolis

We began our analysis by applying the brute‑force Metropolis algorithm to estimate the ground‑state energy of the harmonic oscillator. This provided a baseline against which more advanced sampling strategies could later be compared. [](#figure:vmc-harmonic-grid-analytic) shows grid-search results for the variational parameter $\alpha$ using the analytical expression for the local energy given in [](#equation:local-energy-noninteracting-analytic). To assess how system size and dimensionality influence the variational landscape, simulations were performed for particle numbers $N=1, 10, 100$ and $500$ particles and $d=1,2$ and $3$ spatial dimensions. 

In all cases, the energy curves display a clear minimum near $\alpha \approx 0.5$ corresponding to the expected energy $\bar{E}_\alpha \approx 0.5 dN$. This confirms the linear scaling of the ground-state energy with both particle number and dimension, in agreement with the exact expression in [](#equation:ground-state-energy-noninteracting).

An observed trend in our simulations is that that the acceptance rate decreases as the particle number $N$ increases. Increasing $N$ effectively raises the dimensionality of the configuration space in which proposed moves are made in the Metropolis algorith. For a fixed step size, the displacement of the proposed moves in higher dimensions become increasingly larger relative to the width of the wavefunction, leading to a reduced acceptance rate. To maintain stable acceptance rate and ensure numerical stability for increasing $N$, we adopt a scaled step size $s/\sqrt{N}$ in our simulations, where $s$ is a baseline displacement parameter.

```{figure} figures/vmc_bose_harmonic_grid_analytic.pdf
:label: figure:vmc-harmonic-grid-analytic
:alt: vmc-harmonic-grid-analytic
:align: center

Grid search energy estimates for the harmonic oscillator obtained using brute-force Metropolis algorithm with analytical local energy for $N=1,10,100,500$ particles in $d=1,2,3$ spatial dimensions. All simulations use $\num{1e5}$ Monte Carlo cycles and a scaled step size $s = 1.0 / \sqrt{N}$.
```

We also employed finite difference derivates to compute the logarithmic Laplacian of the trial wavefunction in the local-energy expression [](#equation:local-energy-noninteracting). As shown in [](#figure:vmc_bose_harmonic_grid_analytic_numerical), a random-walk Metropolis sampler using this numerical approach produces energy curves that closely match those obtained with the analytic expression, albeit at a higher computational cost. Benchmarking the computation times shows that the analytic approach is approximately twice as fast as the numerical one. The second-order finite difference derivation [](#equation:second-order-finite-difference) requires two evaluations of the wavefunction $\Psi_T$ in each coordinate direction, and thus scales as $O(2dN)$ operations.  The analytic expression, in comparison, scales as $O(1)$.

```{figure} figures/vmc_bose_harmonic_grid_analytic_numerical.pdf
:label: figure:vmc-harmonic-grid-analytic-numerical
:alt: vmc-harmonic-grid-analytic-numeric
:align: center

Comparison of grid search energy estimates for the harmonic oscillator obtained using brute-force Metropolis sampler with analytic and numerical local-energy evaluations for $N=100$ particles in $d=3$ spatial dimensions. The simulation uses $\num{1e6}$ Monte Carlo cycles and step size $s = 1.0 / \sqrt{N}$.
```

### Parameter Grid Search using Metropolis Importance Sampling

We next applied the Metropolis algorithm with importance sampling to estimate the ground‑state energy of the harmonic oscillator. [](#figure:vmc-harmonic-importance-time-step-dependence) shows how the estimated energy depends on the time step $\Delta t$. For relatively high $\Delta t$, the proposed moves according to [](#equation:langevin-discretized) become large. This effectively drops the acceptance rate, making causing a frozen walker in the Metropolis algorithm. Consequently, the energy estimates become noisy and biased. Conversely, when $\Delta t$ is too small, the proposed moves become small. This causes the acceptance rate to approach unity, resulting in unstable energy estimates due to slow exploration of the configuration space.

```{figure} figures/vmc_bose_harmonic_grid_importance_analytic_time_step_dependence.pdf
:label: figure:vmc-harmonic-importance-time-step-dependence
:alt: vmc-harmonic-grid-importance-analytic
:align: center

Grid search energy estimates for the harmonic oscillator obtained using the Metropolis algorithm with importance sampling and analytic local energy, computed for $N=500$ particles in $d=3$ spatial dimensions across different time steps $\Delta t$. All simulations use $\num{1e4}$ Monte Carlo cycles.
```

[](#figure:vmc-harmonic-comparison-brute-importance) ompares the energy estimates obtained with the brute‑force Metropolis algorithm and with importance sampling. The brute‑force estimates are noticeably noisier and exhibit a stronger bias. For our particular experiment, the mean acceptance rate for the brute‑force sampler is approximately $0.73$, whereas importance sampling achieves an acceptance rate close to unity. This indicates that importance sampling reaches the equilibrium distribution more efficiently and therefore requires fewer Monte Carlo cycles to produce stable energy estimates.

```{figure} figures/vmc_bose_harmonic_grid_analytic_numerical.pdf
:label: figure:vmc-harmonic-comparison-brute-importance
:alt: vmc-harmonic-comparison-brute-importance
:align: center

Comparison of grid search energy estimates for the harmonic oscillator obtained using brute-force Metropolis sampler and importance sampling, both employing analytic local-energy evaluations. The simulation is carried out for $N=100$ particles in $d=3$ spatial dimensions, and uses $\num{1e5}$ Monte Carlo cycles. The brute-force approach employs a time step $s = 0.01 / \sqrt{N}$, while the importance sampling approach uses a time step $\Delta t = 0.05 / \sqrt{N}$.
```

### Parameter Optimization using Gradient Descent

To estimate the ground-energy for the harmonic oscillator, we used [](#equation:energy-derivatives) to calculate the energy derivative and applied the steepest-descent method to optimize the variational parameter $\alpha$. [](#table:vmc-harmonic-parameter-optimization) summarizes the results for systems with $N=10, 100, 500$ particles in $3$ dimensions. In all cases, the optimized values of $\alpha$ lie very to the exact value $\alpha = 1/2$, showing that the optimization procedure reliably identifies the correct minimum. The corresponding energy per particle $\braket{E}/N$ is close to the exact value $2/3$, with only small statistical deviations. The estimated biases are also statistically insignificant, indicating that the bootstrap resampling is stable and that the simulations have reached equilibrium.

A recurring observation in our simulations is that the stochastic nature of variational Monte Carlo causes the estimated parameters to fluctuate around the local minimum rather than converging smoothly toward it. This behaviour makes the optimization trajectory noisy and highlights the need for stabilization techniques to counteract the inherent noise in the gradient estimate.

To stabilize the noisy gradient estimates inherent in variational Monte Carlo, we adopted an optimizer combining a warmup–cosine-decay learning‑rate schedule with adaptive moment estimation (ADAM) and global-norm clipping. The warmup phase prevents unstable early updates, while the gradual cosine decay reduces step sizes as the optimization approaches the minimum. ADAM's adaptive scaling further smooths stochastic fluctuations, and gradient clipping suppresses rare large updates.

:::{table} Optimized variational energies $\braket{E}$ and corresponding parameter $\alpha$ for the harmonic oscillator with $N$ particles in $d=3$ spatial dimensions. The energies, variances and biases are estimated using bootstrap resampling over a Metropolis importance sampling run of $2^{20}$ Monte Carlo cycles with a time step $\Delta t = 0.05 / \sqrt{N}$.
:label: table:vmc-harmonic-parameter-optimization
:align: center

| $N$ | $\alpha$ | $\braket{E}/N$ | $\operatorname{var}(E)/N$ | $\operatorname{bias}(E)/N$ |
|---|---|---|---|---|
| $\num{10}$ | $\num{0.5016}$ | $\num{1.5000}$ | $\num{4.998e-07}$ | $\num{-2.376e-07}$ |
| $\num{100}$ | $\num{0.5004}$ | $\num{1.5033}$ | $\num{1.114e-04}$ | $\num{-1.466e-07}$ |
| $\num{500}$ | $\num{0.5003}$ | $\num{1.4969}$ | $\num{1.588e-04}$ | $\num{3.990e-07}$ |
:::

## Interacting Bose Gas

This section presents simulation results obtained using variational Monte Carlo to estimate the ground‑state energy of a repulsively interacting Bose gas.

### Parameter Grid Search using Metropolis Importance Sampling

To estimate the ground-state energy for the interacting Bose gas, we performed grid search over $\alpha$ while keeping the anisotropy fixed at $\beta = \gamma = 2.82843$ and the characteristic length at $a = 0.0043$. [](#figure:vmc-repulsive_grid-importance-analytic) shows that energy curves attain their minima for $\alpha < 1/2$. As the particles number increses, the optimal $\alpha$ shift to smaller values, and the corresponding per-particle energy rises. 

A lower $\alpha$ indicates that the Gaussian factor in the wavefunction [](#equation:trial-wavefunction) becomes broader, effictelvely producing a weaker confinement within the elliptic trap. This describes an expansion of the condensate due to repulsive interactions. The increase in total energy indicates that the interaction energy grows more rapidly than the reduction in kinetic energy gained from this expansion.

```{figure} figures/vmc_bose_repulsive_grid_importance_analytic.pdf
:label: figure:vmc-repulsive_grid-importance-analytic
:alt: vmc-repulsive_grid-importance-analytic
:align: center

Grid search energy estimates for the interacting Bose gas obtained using Metropolis importance sampling with analytic local energies for $N=10, 100, 500$ particle. All simulations use $100,000$ Monte Carlo cycles and a time step $\Delta t = 0.05$.
```

### Parameter Optimization using Gradient Descent

Following the same optimization procedure as for the harmonic oscillator, we estimated the ground-state energy of the repulsively Bose gas by combining variational Monte Carlo estimation and steepest descent. The results for systesms with $N=10, 100, 500$ particles in $3$ dimensions are summarized in [](#table:vmc-repulsive-parameter-optimization). 

In contrast to the non-interacting case, the optimized values of $\alpha$ decreases systematically with increasing particle number, reflecting the expected expansion of the condensate driven by repulsive Jastrow correlations. The corresponding energy per particle $\braket{E}/N$ rises with $N$, consistent with the growing contribution of the interaction energy. The statistical variances reamin small relative to the total energy, and the estimated biases are negligible, indicating that the bootstrap analysis is stable and that the simulation have reached equilibrium.

:::{table} Optimized variational energies $\braket{E}$ and corresponding parameter $\alpha$ for the repulsively interacting Bose gas with $N$ particles. The energies, variances and biases are estimated using bootstrap resampling over a Metropolis importance sampling run of $2^20$ Monte Carlo cycles with a time step $\Delta t = 0.05 / \sqrt{N}$.
:label: table:vmc-repulsive-parameter-optimization
:align: center

| $N$ | $\alpha$ | $\braket{E}/N$ | $\operatorname{var}(E)/N$ | $\operatorname{bias}(E)/N$ |
|---|---|---|---|---|
| $\num{10}$ | $\num{0.4937}$ | $\num{2.4330}$ | $\num{2.596e-04}$ | $\num{6.281e-06}$ |
| $\num{100}$  | $\num{0.4530}$ | $\num{2.6871}$ | $\num{2.051e-04}$ | $\num{-5.600e-06}$ |
| $\num{500}$  | $\num{0.3676}$ | $\num{3.5641}$ | $\num{1.667e-04}$ | $\num{1.105e-05}$ |
:::

### One-body Densities

Having optimized the Gaussian confinement parameter $\alpha$, we estimated the corresponding one-body densities for both the non-interacting harmonic oscillator and the repulsively interacting bose gas. [](#figure:vmc-onebody-density) shows how repulsive Jastrow correlations shape the one-body densities as the particle number $N$ increases. For $N = 10$, the interacting density has a sharper peak near the origin compared with the harmonic oscillator. This reflects the dominance of the external confinement over the repulsive interactions. At $N = 100$, the repulsive interactions become strong enough to counterbalance the trap, producing a density profile that more closely resembles the non-interacting system. For $N = 500$, the interacting distribution flatten and develops a longer tail. This reflects a regime in which the interaction energy is sufficiently large to expand the gas outward and reduce the influence of the elliptical trap. 

On a sidenote, the one-body density for $N = 500$ is noticeably more noisy. This indicates that number Monte Carlo cycles used in the simulation lies near the lower threshold required for reliable sampling at large $N$. As the particle number grows, the configuration space becomes higher-dimensional and the repulsive correlations slow down the exploration of that space. Consequently, more cycles are needed to reduce autocorrelation and achieve smooth, well-converged density estimates.

```{figure} figures/vmc_bose_onebody_density.pdf
:label: figure:vmc-onebody-density
:alt: vmc-onebody-density
:align: center

Normalized one-body densities as a function of radial coordinate $r$ of both the non-interacting harmonic oscillator and the repulsively interacting bose gas, shown for several particle numbers $N$. All densities are estimated using Metropolis importance sampling with $\num{1e6}$ cycles.
```

# Conclusions

To assess the efficiency of different sampling strategies in variational Monte Carlo, we evaluated the ground‑state energy of the harmonic oscillator using both the brute‑force Metropolis algorithm and the importance‑sampling variant based on Langevin dynamics. The two approaches differ primarily in how proposed moves are generated: brute‑force Metropolis relies on symmetric random displacements, while importance sampling incorporates drift terms that guide the walker toward regions of high probability density. This distinction has direct consequences for acceptance rates, sampling efficiency, and ultimately the stability of the energy estimates.

Using variational Monte Carlo combined with steepest‑descent optimization, we determined the optimal Gaussian confinement parameter for both the non‑interacting harmonic oscillator and the repulsively interacting Bose gas. For the harmonic oscillator, the method successfully recovered the expected variational minimum, demonstrating that the implementation reliably identifies the correct local energy minima. For the interacting system, the optimization revealed the characteristic broadening of the condensate induced by repulsive correlations, reflected in a systematic shift of the optimal confinement parameter to smaller values as the particle number increases. The corresponding rise in ground‑state energy is consistent with the growing contribution of interaction energy, which outweighs the reduction in kinetic energy associated with the expanded density profile. Together, these results confirm that the variational framework captures both the exact non‑interacting limit and the qualitative physical behavior of repulsively interacting Bose gases.

# Appendix

## Code Repository

The Python source code used for this project is available at [https://github.com/semapheur/fys4411](https://github.com/semapheur/fys4411).

## Local Energy in the Non-interacting Case

Defining the one-particle factor $\phi(\mathbf{r}_i) := \exp(-\alpha r_i^2)$, where $r_i = |\mathbf{r}_i|$, the trial wave function [](#equation:trial-wavefunction-noninteracting) can be written as

$$
  \Psi_T (\mathbf{R}) = \prod_{i=1}^N e^{-\alpha r_i^2}.
$$

Since the gradient operator $\nabla_i := \nabla_{\mathbf{r}_i}$ acts only on the coordinate $\mathbf{r}_i$ for each $i$, the Laplacian of $\Psi_T$ becomes

$$
  \nabla_i^2 \Psi_T (\mathbf{R}) = \left(\prod_{j\neq i} \phi(\mathbf{r}_j) \right) \nabla_i^2 \phi(\mathbf{r}_i).
$$

Dividing by $\Psi_T$ yields the identity

$$
  \frac{\nabla_i^2 \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)}
$$

Substituting this into [](#equation:local-energy), we obtain the local energy

$$
\begin{align*}
  E_L (\mathbf{R}) =& \frac{1}{\Psi_T (\mathbf{R})} \left(\sum_{i=1}^N -\frac{\hbar^2}{2m} \nabla_i^2 \Psi(\mathbf{R}) + V_\text{ext} (\mathbf{r}_i) \Psi_T (\mathbf{R}) \right) \\
  =& \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)} + V_\text{ext} (\mathbf{r}_i) \right) \label{equation:local-energy-spherical-noninteracting}
\end{align*}
$$

To obtain an analytic expression for $E_L$ in a $d$-dimensional system, we compute the logarithmic Laplacian of $\phi(\mathbf{r})$. The logarithmic gradient of $\phi(\mathbf{r})$ is

$$
\label{equation:gaussian-factor-spherical-log-gradient}
  \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} = \nabla \ln\left(e^{-\alpha r^2} \right) = -2\alpha\mathbf{r},
$$

and the logarithmic Laplacian is

$$
\begin{align*}
  \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} =& \nabla^2 (\ln[\phi(\mathbf{r}]) + |\nabla(\ln[\phi(\mathbf{r})])|^2 \\
  =& -2\alpha \underbrace{\nabla\cdot\mathbf{r}}_{=d} + |-2\alpha \mathbf{r}|^2 = -2d\alpha + 4\alpha^2 r^2
\end{align*}
$$

Substituting this, together with the sperical trap potential

$$
  V_\text{ext} (\mathbf{r}_i) = \frac{1}{2}m\omega_\text{ho}^2 r_i^2,
$$

into [](#equation:local-energy-spherical-noninteracting), we find

$$
\begin{align*}
  E_L (\mathbf{R}) =& \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} (4\alpha^2 r_i^2 - 2d\alpha) + \frac{1}{2} m\omega_\text{ho}^2 r_i^2 \right) \\
  =& \sum_{i=1}^N \left(\frac{d\hbar^2 \alpha}{m} - \frac{2\hbar^2 \alpha^2}{m} r_i^2 + \frac{1}{2} m\omega_\text{ho}^2 r_i^2 \right) \\
  =& \frac{d\hbar^2 \alpha N}{m} + \sum_{i=1}^N \left(\frac{1}{2} m \omega_\text{ho}^2 - \frac{2\hbar^2 \alpha^2}{m} \right) r_i^2
\end{align*}
$$

## Logarithmic Gradient of the Trial Wavefunction

To derive the logarithmic gradient of the trial wavefunction [](#equation:trial-wavefunction), we first rewrite it in the form

$$
\label{equation:trial-wavefunction-interacting}
  \Psi_T (\mathbf{R}) = \left(\prod_{i=1}^N \phi(\mathbf{r}_k) \right) \exp\left(\sum_{j < k} u(r_{kj}) \right),\; r_{kj} = |\mathbf{r}_k - \mathbf{r}_j|
$$

where $\phi(\mathbf{r}_k) = g(\alpha,\beta,\mathbf{r}_i)$ are the Gaussian single-particle factors, and $u(r_{kj}) = \ln[f(r_{kj})]$ are the logarithmic Jastrow correlation factors. Introduing the shorthand,

$$
  A(\mathbf{R}) = \prod_{i=1}^N \phi(\mathbf{r}_i),\quad B(\mathbf{R}) = \exp\left(\sum_{j < k} u(r_{jk}) \right), 
$$

we can factor $\Psi_T$ neatly as

$$
\label{equation:trial-wavefunction-refactor}
  \Psi_T (\mathbf{R}) = A(\mathbf{R}) B(\mathbf{R})
$$

Taking the gradient and applying the product rule yields

$$
  \nabla_k \Psi_T  = (\nabla_k A) B + A \nabla_k B
$$

The first gradient $\nabla_k A$ is given by

$$
  \nabla_k A(\mathbf{R}) = \nabla_k \left(\prod_{i=1}^N \phi(\mathbf{r}_i) \right) = \left(\prod_{i\neq k} \phi(\mathbf{r}_i) \right) \nabla_k \phi(\mathbf{r}_k),
$$

because only the factor $\phi(\mathbf{r}_k)$ depends on $\mathbf{r}_k$. To calculate the second gradient $\nabla_k B$, we first substitute $U(\mathbf{R}) = \sum_{j < k} u(r_{ij})$, such that $B = e^U$. The chain rule yields $\nabla_k B = B \nabla_k U$, hence

$$
  \nabla_k B(\mathbf{R}) = B(\mathbf{R}) \sum_{j\leq k} \nabla_k u(r_{jk})
$$

Combining the terms leads to

$$
\begin{align*}
  \nabla_k \Psi_T (\mathbf{R}) =& \exp\left(\sum_{j < m} u(r_{jm}) \right) \left(\prod_{i\neq k} \phi(\mathbf{r}_i) \right) \nabla_k \phi(\mathbf{r}_k) \\
  &+ \left(\prod_{i=1}^N \phi(\mathbf{r}_i) \right) \exp\left(\sum_{j < m} u(r_{jm}) \right) \sum_{l\neq k} \nabla_k u(r_{kl}) \\
\end{align*}
$$

Factoring out $\Psi_T$, the logarithmic gradient can be written as

$$
\label{equation:wavefunction-log-gradient}
  \frac{\nabla_k \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + \sum_{j\neq k} \nabla_k u(r_{kj})
$$

To derive an analytic expression, we first calculate the logarithmic gradient of $\phi(\mathbf{r})$, given by

$$
\label{equation:gaussian-factor-log-gradient}
  \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} = \nabla \ln\left(\exp[-\alpha(x_i^2 + y_i^2 + \beta z_i^2)] \right) = -2\alpha(x \unitvec{x} + y \unitvec{y} + \beta z \unitvec{z})
$$

To evaluate $\nabla_k u(r_{kj})$, we apply the chain rule, yielding $\nabla_k u(r_{kj}) = u' (r_{kj}) \nabla_k r_{kj}$. Introducing $\mathbf{r} := \mathbf{r}_k - \mathbf{r}_j$, such that $r_{kj} = |\mathbf{r}|$, we get

$$
  \nabla_k r_{kj} = \nabla_k (\mathbf{r} \cdot \mathbf{r})^{1/2} = \frac{1}{2} (\mathbf{r}\cdot\mathbf{r})^{-1/2} \nabla_k (\mathbf{r}\cdot\mathbf{r})
$$

Since

$$
  \nabla_k (\mathbf{r} \cdot \mathbf{r}) = \nabla_k \left(\sum_{\alpha=1}^3 r_\alpha^2 \right) = 2\mathbf{r}
$$

we obtain

$$
  \nabla_k r_{kj} = \frac{1}{2}\frac{1}{|\mathbf{r}|}2\mathbf{r} = \frac{\mathbf{r}}{|\mathbf{r}|} = \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}}
$$

Hence,

$$
  \nabla_k u(r_{jk}) = u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}}.
$$

Derivating $u(r)$, we get for $a > -1$

$$
\label{equation:jastrow-first-derivative}
  u'(r) = \frac{\drm}{\drm r} \ln[f(a, r)] = \frac{\drm}{\drm r} \ln\left(1 - \frac{a}{r} \right) = \frac{a}{r^2} \frac{1}{1 - a/r} = \frac{a}{r^2 - ar}
$$

Substituting this and [](#equation:gaussian-factor-log-gradient) into [](#equation:wavefunction-log-gradient), leads to

$$
\label{equation:wavefunction-log-gradient-analytic}
  \frac{\nabla_k \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = -2\alpha(x_k \unitvec{x} + y_k \unitvec{y} + \beta z_k \unitvec{z}) + a \sum_{j\neq k} \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}^2 (r_{kj} - a)}
$$

## Logarithmic Laplacian of the Trial Wavefunction

To derive the logarithmic Laplacian of $\Psi_T$, we define

$$
  \mathbf{F}_T := \frac{\nabla_k \Psi_T}{\Psi_T}
$$

such that

$$
  \nabla_k^2 \Psi_T = \nabla_k \cdot (\nabla_k \Psi_T) = \nabla_k \cdot (\Psi_T \mathbf{F}_k)
$$

Using the product rule for divergence, we get

$$
\begin{align*}
  \nabla_k (\Psi_T \mathbf{F}_k) =& \underbrace{(\nabla_k \Psi_T)}_{=\Psi_T \mathbf{F}_k} \cdot \mathbf{F}_k + \Psi_T (\nabla_k \cdot \mathbf{F}_k) \\
  =& \Psi_T |\mathbf{F}_k|^2 + \Psi_T (\nabla_k \cdot \mathbf{F}_k)
\end{align*}
$$

Dividing by $\Psi_T$, we obtain

$$
\label{equation:laplacian-identity-simplified}
  \frac{\nabla_k^2 \Psi_T}{\Psi_T} = |\mathbf{F}_k|^2 + \nabla_k \mathbf{F}_k
$$

Letting

$$
  \mathbf{A}_k = \frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)},\quad \mathbf{B}_k = \sum_{j\neq k} u' (r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}}
$$

We have

$$
  |\mathbf{F}_k|^2 = |\mathbf{A}_k + \mathbf{B}_k|^2 = |\mathbf{A}_k|^2 + 2\mathbf{A}_k \cdot \mathbf{B}_k + |\mathbf{B}_k|^2
$$

with

$$
  |\mathbf{B}_k|^2 = \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)\cdot(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj}} u'(r_{ki}) u'(r_{kj})
$$

The divergence term in [](#equation:laplacian-identity-simplified) is given by the product rule

$$
  \nabla_k \cdot \mathbf{F}_k = \nabla_k \cdot \mathbf{A}_k + \nabla_k \cdot \mathbf{B}_k,
$$

where the first term evaluates to

$$
\begin{align*}
  \nabla_k \cdot \mathbf{A}_k =& \nabla_k \cdot \left(\frac{\nabla_k \phi}{\phi} \right) = \nabla_k \left(\frac{1}{\phi} \right) \cdot \nabla_k \phi + \frac{1}{\phi} \nabla_k^2 \phi \\
  =& -\frac{1}{\phi^2} (\nabla_k \phi) \cdot (\nabla_k \phi) + \frac{1}{\phi} \nabla_k^2 \phi  = \frac{\nabla_k^2 \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} - \left|\frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} \right|^2
\end{align*}
$$

For the second term, we define the substitutions

$$
  \mathbf{x} := \mathbf{r}_k - \mathbf{r}_j,\quad x := r_{kj},\quad v(x) = \frac{u'(x)}{x},
$$

so that

$$
  u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}} = v(x) \mathbf{x}
$$

Since $\nabla_\mathbf{x} = \nabla_k$, the divergence product rule yields

$$
\begin{align*}
  \nabla_\mathbf{x} \cdot (v(x) \mathbf{x}) =& (\nabla_\mathbf{x} v) \cdot \mathbf{x} + v(x) \underbrace{\nabla_\mathbf{x} \cdot \mathbf{x}}_{=3} = v'(x) \underbrace{(\nabla_\mathbf{x} x)}_{=\mathbf{x}/x} \cdot \mathbf{x} + 3v(x) \\
  =& x v'(x) + 3v(x)
\end{align*} 
$$

Applying the quotient rule to $v(x) = u'(x)/x$

$$
  v'(x) = \frac{u''(x)x - u'(x)}{x^2},
$$

and substituting, yields

$$
  \nabla_\mathbf{x} \cdot (v(x) \mathbf{x}) = \frac{u''(x) - u'(x)}{x} + \frac{3u'(x)}{x} = u''(x) + \frac{2}{x} u'(x)
$$

Substituting back, we get

$$
\begin{align*}
  \nabla_k \cdot \left(u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}} \right) = u'' (r_{kj}) + \frac{2}{r_{kj}} u'(r_{kj})
\end{align*} 
$$

and summing over $j \neq k$,

$$
  \nabla_k \cdot \mathbf{B}_k = \sum_{j\neq k} \left(u'' (r_{kj}) + \frac{2}{r_{kj}} u'(r_{kj}) \right).
$$

Combining the terms, the logarithmic Laplacian of $\Psi_T$ takes the form

$$
\label{equation:wavefunction-logarithmic-laplacian}
\begin{split}
  \frac{1}{\Psi_T} \nabla_k^2 \Psi_T (\mathbf{r}) =& \frac{\nabla_k^2 \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + 2\frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} \left(\sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_j)}{r_{kj}} u'(r_{kj}) \right) \\
  &+ \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)\cdot(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj}} u'(r_{ki}) u'(r_{kj}) \\
  &+ \sum_{j\neq k} \left(u'' (r_{kj}) + \frac{2}{r_{kj}} u' (r_{kj}) \right)
\end{split}
$$

To obtained a closed analytic form of this expression, we compute the logarithmic gradient and Laplacian of $\phi$. Applying [](#equation:gaussian-factor-log-gradient), we find

$$
\begin{align*}
  \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} =& \nabla^2 (\ln[\phi(\mathbf{r})]) + |\nabla (\ln[\phi(\mathbf{r})])|^2 \\
  =& -2\alpha(2 + \beta) + 4\alpha^2 (x^2 + y^2 + \beta^2 y^2)
\end{align*}
$$

Using [](#equation:jastrow-first-derivative), with $g(r) = r^2 - ar$ and $g'(r) = 2r - a$, the second derivative of the logarithmic Jastrow factor becomes

$$
  u''(r) = -a\frac{g'(r)}{g(r)^2} = \frac{a^2 - 2ar}{(r^2 - ar)^2}
$$

Inserting the calculated derivatives into [](#equation:wavefunction-logarithmic-laplacian) and simplifying, we finally arrive at

$$
\begin{align*}
  \frac{1}{\Psi_T (\mathbf{R})} \nabla_k^2 \Psi_T (\mathbf{R}) =& -2\alpha(2 + \beta) + 4\alpha^2 (x_k^2 + y_k^2 + \beta^2 z_k^2) \\
  &- 4\alpha a \sum_{j\neq k} \frac{(x_k - x_j)x_k + (y_k - y_j)y_k + \beta(z_k - z_j)z_k}{r_{kj} (r_{kj}^2 - ar_{kj})} \\
  &+ a^2 \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)\cdot(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj} (r_{ki}^2 - ar_{ki})(r_{kj}^2 - ar_{kj})} \\
  &- a^2 \sum_{j\neq k} \frac{1}{r_{kj}^2 (r_{kj} - a)^2}
\end{align*}
$$

## One-body Density in the Non-Interacting System

For a spherical trap with $\beta = 1$ in a non-interacting system, the normalized trial function takes the form

$$
  \Psi_T (\mathbf{r}_1,\dots,\mathbf{r}_N; \alpha) = \prod_{i=1}^N A e^{-er_i^2}
$$

with probability density

$$
  |\Psi_T|^2 = |A|^{2N} \prod_{i=1}^N e^{-2\alpha r_i^2}
$$

The one-body function $\phi(\mathbf{r}) = A \exp(-\alpha r^2)$ must satisfy

$$
  \int_{\R^3} |\phi(\mathbf{r})|^2 \;\drm\mathbf{r} = |A|^2 \int_{\R^3} e^{-2\alpha r^2} \;\drm\mathbf{r} = 1
$$

Using

$$
  \int_{\R^3} e^{-2\alpha r^2} \;\drm\mathbf{r} = \left(\frac{\pi}{2\alpha} \right)^{3/2},
$$

we obtain the normalization constant

$$
  |A|^2 = \left(\frac{2\alpha}{\pi}\right)^{3/2}.
$$

The one-body density is then given by

$$
\begin{align*}
  \rho(\mathbf{r}) =& N \int_{\R^{3(N-1)}} |\Psi_T (\mathbf{r},\mathbf{r}_2, \dots, \mathbf{r}_N)|^2 \;\drm\mathbf{r}_2 \cdots \drm\mathbf{r}_N \\
  =& N|\phi(\mathbf{r})|^2 \prod_{j=2}^N \underbrace{\left(\int_{\R^3} |\phi(\mathbf{r}_j)|^2 \right)}_{=1} \;\drm \mathbf{r}_j = N \left(\frac{2\alpha}{\pi} \right)^{3/2} e^{-2\alpha r^2}
\end{align*}
$$

## Energy Derivative Formula

To derive the formula for the energy derivative [](#equation:energy-derivative) with respect to a parameter $\alpha_k$, we apply the quotient rule to $E(\boldsymbol{\alpha}) = N/D$ defined in [](#equation:hamiltonian-expectation). This gives

$$
\label{equation:quotient-rule}
  \frac{\partial E(\boldsymbol{\alpha})}{\partial\alpha_k} = \frac{(\partial_{\alpha_k} N) D - N(\partial_{\alpha_k} D)}{D^2} = \frac{(\partial_{\alpha_k} N)}{D} - E(\boldsymbol{\alpha}) \frac{\partial_{\alpha_k} D}{D},
$$

The numerator partial derivative is

$$
\begin{align*}
  \partial_{\alpha_k} N =& \frac{\partial}{\partial\alpha_k} \int_{\R^{3N}} \Psi_T(\mathbf{R}; \boldsymbol{\alpha}) \hat{H}\Psi_T (\mathbf{R}; \boldsymbol{\alpha}) \;\drm \mathbf{R} \\
  =& \int [(\partial_{\alpha_k} \Psi_T) \hat{H}\Psi_T + \Psi_T \hat{H} (\partial_{\alpha_k} \Psi_T)] \;\drm\mathbf{R}
\end{align*}
$$

Since $\hat{H}$ is Hermitian, we get

$$
  \int \Psi_T \hat{H} (\partial_{\alpha_k} \Psi_T) \;\drm\mathbf{R} = \int (\partial_{\alpha_k} \Psi_T) \hat{H} \Psi_T \;\drm\mathbf{R}
$$

Substituting the local energy $E_L$ defined in [](#equation:local-energy) results in

$$
  \partial_{\alpha_k} N = 2 \int (\partial_{\alpha_k} \Psi_T) \hat{H} \Psi_T \;\drm \mathbf{R} = 2 \int |\Psi_T|^2 \left(\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} \right) E_L \;\drm\mathbf{R}
$$

The denominator derivative is

$$
\begin{align*}
  \partial_{\alpha_k} D =& \frac{\partial}{\partial\alpha_k} \int_{\R^{3N}} |\Psi_T (\mathbf{R}; \boldsymbol{\alpha})|^2 \;\drm\mathbf{R} \\
  =& 2 \int \Psi_T (\partial_{\alpha_k} \Psi_T) \;\drm\mathbf{R} = 2 \int |\Psi_T|^2 \left(\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} \right)
\end{align*}
$$

Inserting into the quotient rule [](#equation:quotient-rule) yields

$$
\begin{align*}
  \frac{\partial E}{\partial\alpha_k} =& 2 \int \left(\frac{|\Psi_T|^2}{\int |\Psi_T|^2 \;\drm\mathbf{R}}\right) \left(\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} \right) E_L \;\drm\mathbf{R} \\
  &- 2 E(\boldsymbol{\alpha}) \int \left(\frac{|\Psi_T|^2}{\int |\Psi_T|^2 \;\drm \mathbf{R}} \right) \left(\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} \right) \;\drm\mathbf{R}
\end{align*}
$$

Inserting the probability density $P_{\alpha,\beta}$ defined in [](#equation:wavefunction-pdf), we identify the expectation values

$$
  \Braket{\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} E_L} = \int P_{\boldsymbol{\alpha}} (\mathbf{R}) \left(\frac{\partial_\gamma \Psi_T (\mathbf{R}; \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}; \boldsymbol{\alpha})} \right) E_L (\mathbf{R}; \boldsymbol{\alpha}) \;\drm\mathbf{R}
$$

and

$$
  \Braket{\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T}} = \int P_{\boldsymbol{\alpha}} (\mathbf{R}) \left(\frac{\partial_{\alpha_k} \Psi_T (\mathbf{R}; \boldsymbol{\alpha})}{\Psi_T (\mathbf{R}; \boldsymbol{\alpha})} \right) \;\drm\mathbf{R}
$$

Also, since $E(\boldsymbol{\alpha}) = \braket{E_L}$ as defined in [](#equation:variation-energy-expectation), we finally arrive at

$$
  \frac{\partial E}{\partial\alpha_k} = 2\left(\Braket{\frac{\partial_{\alpha_k} \Psi_T}{\Psi_T} E_L} - \Braket{\frac{\partial_{\alpha_k}}{\Psi_T}} \braket{E_L} \right)
$$
