---
title: FYS4411 - Project 1
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    #template: ../../report_template
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
---

# Theory

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
  \Psi_T (\mathbf{R}) = \prod_{i=1}^N e^{-\alpha r_i^2},\; r_i = |\mathbf{r}_i| = \exp\left(\alpha \sum_{i=1}^N r_i^2 \right).
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
  E_L (\mathbf{R}) = \frac{d\hbar^2 \alpha N}{m} + \sum_{i=1}^N \left(\frac{1}{2} m\omega_\text{ho}^2 - \frac{2\hbar^2 \alpha^2}{m} \right) r_i^2.
$$

The sum vanishes when

$$
  \alpha = \frac{m\omega_\text{ho}}{2\hbar},
$$

which corresponds to the exact ground-state width of the harmonic-oscillator. In this case, the local energy becomes a constant,

$$
  E_L = \frac{d}{2} N\hbar \omega_\text{ho},
$$

which coincides with the exact ground state-energy of $N$ non-interacting bosons in a spherical harmonic trap.

##### Drift Force

In the non-interacting case, the drift force used for importance sampling is given by

$$
  \mathbf{F}_i = \frac{2\nabla_i \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = -4\alpha \mathbf{r}_i,
$$

where we have used [](#equation:gradient).

#### Interacting Case

In the interacting case $a > 0$, the local energy takes the form

$$
  E_L (\mathbf{R}) = \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} \frac{\nabla_i^2 \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} + V_\text{ext}(\mathbf{r}_i) \right) + \sum_{i=1}^N \sum_{j=i+1}^N V_\text{int} (\mathbf{r}_i, \mathbf{r}_j)
$$

To evaluate this expression, we begin by rewriting $\Psi_T$ in [](#equation:trial-wavefunction) as

$$
\label{equation:trial-wavefunction-interacting}
  \Psi_T (\mathbf{R}) = \left(\prod_{i=1}^N \phi(\mathbf{r}_k) \right) \exp\left(\sum_{j < k} u(r_{jk}) \right)
$$

where $\phi(\mathbf{r}_i) = g(\alpha,\beta,\mathbf{r}_i)$, $r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$ and $f(r_{ij}) = e^{u(r_{ij})}$. In this form, the logarithmic Laplacian of $\Psi_T$ becomes

$$
\label{equation:trial-wavefunction-interacting-laplacian}
\begin{split}
  \frac{1}{\Psi_T (\mathbf{R})} \nabla_i^2 \Psi_T (\mathbf{R}) =& -2\alpha(2 + \beta) + 4\alpha^2 (x_i^2 + y_i^2 + \beta^2 z_i^2) \\
  &- 4\alpha a \sum_{j\neq i} \frac{(x_i - x_j)x_k + (y_i - y_j)y_k + \beta(z_i - z_j)z_k}{r_{ij}^2 (r_{ij} - a)} \\
  &+ a^2 \sum_{j\neq i} \sum_{k\neq i} \frac{(\mathbf{r}_i - \mathbf{r}_j)\cdot(\mathbf{r}_i - \mathbf{r}_k)}{r_{ij}^2 r_{ik}^2 (r_{ij} - a)(r_{ik} - a)} \\
  &- a^2 \sum_{j\neq i} \frac{1}{r_{ij}^2 (r_{ij} - a)^2}
\end{split}
$$

In terms of the dimensionless Hamiltonian in [](#equation:hamiltonian-dimensionless), the local energy becomes

$$
  E_L (\mathbf{R}) = \sum_{i=1}^N \frac{1}{2} \left(-\frac{\nabla^2 \Psi_T(\mathbf{R})}{\Psi_T (\mathbf{R})} + x_i^2 + y_i^2 + \gamma^2 z_i^2 \right) + \sum_{i < j} V_\text{int} (|\mathbf{r}_i - \mathbf{r}_j|)
$$

## Variational Monte Carlo Estimation

The ground state energy of the bose gas correlated model can be estimated using variational Monte Carlo methods. The expectation value of the Hamiltonian $\hat{H}$, in the state $\Psi_T (\mathbf{R}; \alpha, \beta)$ is given by

$$
\begin{equation*}
\label{equation:hamiltonian-expectation}
\begin{split}
  E(\alpha, \beta) :=& \braket{H}_{\Psi_T} = \frac{\braket{\Psi_T (\mathbf{R}; \alpha, \beta), \hat{H} \Psi_T (\mathbf{R}, \alpha, \beta)}}{\braket{\Psi(\mathbf{R}, \alpha, \beta), \Psi_T (\mathbf{R}; \alpha, \beta)}} \\
  =& \frac{\int_{\R^{3N}} \Psi_T^* (\mathbf{R}; \alpha, \beta) \hat{H}\Psi_T (\mathbf{R}; \alpha, \beta) \;\drm\mathbf{R}}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}; \alpha, \beta)|^2 \;\drm\mathbf{R}}
\end{split}
\end{equation*}
$$

By the Rayleigh-Ritz principle $E(\alpha, \beta) \geq E_0$, where $E_0$ is the ground-state energy. Thus, we can approximate $E_0$ by minimizing $E(\alpha, \beta)$ over the parameters $(\alpha, \beta)$.

To enable Monte Carlo estimation, we expand the variational energy [](#equation:trial-wavefunction) in terms of the probability density function

$$
\label{equation:wavefunction-pdf}
  P_{\alpha,\beta} (\mathbf{R}) = \frac{|\Psi_T (\mathbf{R}; \alpha, \beta)|^2}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}; \alpha, \beta)|^2 \;\drm\mathbf{R}}
$$

Substituting the local energy [](#equation:trial-wavefunction), the variational energy can be written as the expectation value

$$
\label{equation:variation-energy-expectation}
  E(\alpha, \beta) = \int_{\R^{3N}} P_{\alpha,\beta} (\mathbf{R}) E_L (\mathbf{R}; \alpha, \beta) \;\drm\mathbf{R} = \mathbb{E}_{P_{\alpha, \beta}} (E_L)
$$

Consequently, computing $E(\alpha, \beta)$ reduces to sampling from $P_{\alpha,\beta} (\mathbf{R}) \propto |\Psi_T (\mathbf{R}, \alpha, \beta)|^2$. However, the normalization constant

$$
  Z = \int_{\R^{3N}} |\Psi_T (\mathbf{R}, \alpha,\beta)|^2 \;\d\mathbf{R}
$$

is intractable in high dimensions, making direct sampling from $P_{\alpha,\beta}$ unfeasible. To overcome this, we can employ the Metropolis-Hastings algorithm to construct a Markov chain $\set{\mathbf{R}^{(k)}}_{k\geq 0}$ whose stationary distribution is $P_{\alpha,\beta}$.

Given the current state $\mathbf{R}$, we propose a move $\mathbf{R}' \sim T(\cdot|\mathbf{R})$, where $T$ is a chosen transition kernel. The proposal is accepted with probability

$$
  A(\mathbf{R},\mathbf{R}') = \min\Set{1, \frac{|\Psi_T (\mathbf{R}', \alpha,\beta)|^2 T(\mathbf{R}|\mathbf{R}')}{|\Psi_T (\mathbf{R}, \alpha, \beta)|^2 T(\mathbf{R}' |\mathbf{R})}}
$$

If the transition kernel is symmetric, i.e. $T(\mathbf{R}' | \mathbf{R}) = T(\mathbf{R}|\mathbf{R}')$, the acceptance probability simplifies to

$$
  A(\mathbf{R},\mathbf{R}') = \min\Set{1, \frac{|\Psi_T (\mathbf{R}', \alpha,\beta)|^2}{|\Psi_T (\mathbf{R}, \alpha, \beta)|^2}},
$$

This special case corresponds to the original algorithm by [](article_metropolis_etal_1953), referred to as the *Metropolis algorithm*.

Using this procedure to generate samples $\set{\mathbf{R}_k}_{k=1}^M$ \sim P_{\alpha,\beta}$, the expectation [](#equation:trial-wavefunction) approximates to the empirical mean

$$
\label{equation:energy-estimation}
  E(\alpha, \beta) \approx \frac{1}{M} \sum{k=1}^M E_L (\mathbf{R}^{(k)}, \alpha, \beta)
$$

The Metropolis algorithm using a symmetric transition kernel can be summarized as follows:

Given parameters $(\alpha, \beta)$:
1. Initialize the configuration $\mathbf{R}^{(0)}$
2. For $k = 0,\dots,M-1$:
    - Propose a new configurations $\mathbf{R}' \sim T(\cdot|\mathbf{R}^{(k)})$
    - Compute the acceptance ratio
$$
  A = \frac{|\Psi_T (\mathbf{R}', \alpha, \beta)|^2}{|\Psi_T (\mathbf{R}^{(k)}, \alpha, \beta)|^2}
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

To improve the efficiency of the Metropolis algorithm, we can apply importance sampling using Langevin molecular dynamics. 
A diffusion process characterized by a time-dependent probability density $P(\mathbf{R}, t)$ on $\R^{3N}$, is given by the Fokker-Planck equation

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

with a probabiliy density of the form $P(\mathbf{R}) \propto |\Psi_T (\mathbf{R})|^2$, we obtain

$$
  \mathbf{F}(\mathbf{R}) = \nabla \ln[|\Psi_T (\mathbf{R})|^2] = 2 \frac{\nabla \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})}
$$

This quantity is known as the quantum force, which is introduces a drift towards regions of the configuration space where $|\Psi_T|^2$ is large. This significantly improves sampling efficiency compared with the Metropolis algorithm, which explores the configuration space through a symmetric random walk. 

The Fokker-Planck equation yields a transition kernel given by the Green's function

$$
  G(\mathbf{R}', \mathbf{R}, \Delta t) = \frac{1}{(4\pi D\Delta t)^{3N/2}} \exp\left(-\frac{[\mathbf{R}' - \mathbf{R} - D\Delta t \mathbf{F}(\mathbf{R})]^2}{4D\Delta t} \right)
$$

The acceptance probability for the Metropolis-Hastings algorithm now becomes

$$
  A(\mathbf{R}, \mathbf{R}') = \min\Set{1, \frac{G(\mathbf{R}, \mathbf{R}', \Delta t) |\Psi_T (\mathbf{R}')|^2}{G(\mathbf{R}', \mathbf{R}, \Delta t) |\Psi_T (\mathbf{R})|^2}}
$$

Introducing

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

## Parameter Optimization

The optimal variational parameters $(\alpha, \beta)$ can be obtained by minimizing the variational energy $E(\alpha, \beta)$. This a can be achieved by using the gradient $\nabla_{\alpha, \beta} E(\alpha, \beta)$ with respect to variational parameters as the objective for an optimization procedure. The energy derivatives can be written as

$$
  \frac{\partial E}{\partial\gamma} = 2 \left( \Braket{\frac{\partial_\gamma \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\mathbf{R}; \alpha, \beta)} E_L (\mathbf{R}; \alpha, \beta)} - \Braket{\frac{\partial_\gamma \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\mathbf{R}; \alpha, \beta)}} \braket{E_L (\mathbf{R}; \alpha, \beta)} \right)
$$

where $\gamma = \alpha, \beta$.

# Results

## Non-Interacting Case

The numerical approach using finite difference derivation to evaluate the Laplacian of $\Psi_T$ requires $O(2nd)$ operations, since it requires calculating the wavefunction twice per coordinate, while the analytic expression scales as $O(1)$.

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

To obtain an analytic expression for $E_L$ in a $d$-dimensional system, we begin by computing the Laplacian $\nabla^2 \phi(\mathbf{r})$. Differentiating once gives

$$
\label{equation:gradient}
  \nabla \phi(\mathbf{r}) = -2\alpha\mathbf{r} e^{-\alpha r^2}.
$$

Taking the divergence and applying the product rule 

$$
  \nabla\cdot [\mathbf{r} \phi(\mathbf{r})] = \phi(\mathbf{r}) \underbrace{\nabla \cdot \mathbf{r}}_{=d} + \mathbf{r} \cdot \nabla\phi(\mathbf{r}) = d\phi(\mathbf{r}) + \mathbf{r}\cdot\nabla\phi(\mathbf{r}),
$$

we obtain

$$
\begin{align*}
  \nabla_i \cdot (-2\alpha\mathbf{r} e^{-\alpha r^2}) =& -2d\alpha e^{-\alpha r^2} + 4\alpha^2 r^2 e^{-\alpha r^2} \\
  =& e^{-\alpha r^2} (4\alpha^2 r^2 - 2d\alpha).
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

## Local Energy in the Interacting Case

To derive the logarithmic Laplacian [](#equation:trial-wavefunction-interacting-laplacian), we substitute

$$
  A(\mathbf{R}) = \prod_{i=1}^N \phi(\mathbf{r}_i),\quad B(\mathbf{R}) = \exp\left(\sum_{j < k} u(r_{jk}) \right), 
$$

such that $\Psi_T$ in [](#equation:trial-wavefunction-interacting) can be factored as

$$
\label{equation:trial-wavefunction-refactor}
  \Psi_T (\mathbf{R}) = A(\mathbf{R}) B(\mathbf{R})
$$

### Logarithmic Gradient of $\Psi_T$

Taking the gradient of [](#equation:trial-wavefunction-refactor) and appling the product rule yields

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

Factoring out $\Psi_T$, we arrive at

$$
  \frac{\nabla_k \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + \sum_{l\neq k} \nabla_k u(r_{kl})
$$

To evaluate $\nabla_k u(r_{kj})$, we apply the chain rule, giving $\nabla_k u(r_{kj}) = u' (r_{kj}) \nabla_k r_kj$. Introducing $\mathbf{r} := \mathbf{r}_k - \mathbf{r}_j$, such that $r_{kj} = |\mathbf{r}|$, we get

$$
  \nabla_k r_{kj} = \nabla_k (\mathbf{r} \cdot \mathbf{r})^{1/2} = \frac{1}{2} (\mathbf{r}\cdot\mathbf{r})^{-1/2} \nabla_k (\mathbf{r}\cdot\mathbf{r})
$$

Since

$$
  \nabla_k (\mathbf{r} \cdot \mathbf{r}) = \nabla_k \left(\sum_{\alpha=1}^3 r_\alpha^2 \right) = 2\mathbf{r}
$$

we obtain

$$
  \nabla_k r_{jk} = \frac{1}{2}\frac{1}{|\mathbf{r}|}2\mathbf{r} = \frac{\mathbf{r}}{|\mathbf{r}|} = \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}}
$$

Hence

$$
  \nabla_k u(r_{jk}) = u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}}
$$

### Logarithmic Laplacian of $\Psi_T$

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

To obtained a closed analytical form of this expression, we compute the logarithmic gradient and Laplacian of $\phi$. The logarithmic gradient of $\phi$ is given by

$$
  \frac{\nabla \phi(\mathbf{r})}{\phi(\mathbf{r})} = \nabla \ln[\phi(\mathbf{r})] = -2\alpha(x \unitvec{x} + y \unitvec{y} + \beta z \unitvec{z}), \mathbf{r} = (x, y, z) \in \R^3
$$

and the logarithmic Laplacian of $\phi$ is given by

$$
\begin{align*}
  \frac{\nabla^2 \phi(\mathbf{r})}{\phi(\mathbf{r})} =& \nabla^2 (\ln[\phi(\mathbf{r})]) + |\nabla (\ln[\phi(\mathbf{r})])|^2 \\
  =& -2\alpha(2 + \beta) + 4\alpha^2 (x^2 + y^2 + \beta^2 y^2)
\end{align*}
$$

Furthermore, we have for $a > 0$

$$
  u'(r) = \frac{\drm}{\drm r} \ln[f(a, r)] = \frac{\drm}{\drm r} \ln\left(1 - \frac{1}{r} \right) = \frac{a}{r^2} \frac{1}{1 - a/r} = \frac{a}{r^2 - ar}
$$

and with $g(r) = r^2 - ar$ and $g'(r) = 2r - a$

$$
  u''(r) = -a\frac{g'(r)}{g(r)^2} = \frac{a^2 - 2ar}{(r^2 - ar)^2}
$$

Inserting the calculated derivatives into [](#equation:wavefunction-logarithmic-laplacian) and simplifying, we arrived at

$$
\begin{align*}
  \frac{1}{\Psi_T (\mathbf{R})} \nabla_k^2 \Psi_T (\mathbf{R}) =& -2\alpha(2 + \beta) + 4\alpha^2 (x_k^2 + y_k^2 + \beta^2 z_k^2) \\
  &- 4\alpha a \sum_{j\neq k} \frac{(x_k - x_j)x_k + (y_k - y_j)y_k + \beta(z_k - z_j)z_k}{r_{kj} (r_{kj}^2 - ar_{kj})} \\
  &+ a^2 \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)\cdot(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj} (r_{ki}^2 - ar_{ki})(r_{kj}^2 - ar_{kj})} \\
  &- a^2 \sum_{j\neq k} \frac{1}{r_{kj}^2 (r_{kj} - a)^2}
\end{align*}
$$

## Energy Derivatives

To derive the expression for the energy derivatives [](#equation:energy-derivative), we apply the quotient rule to $E(\alpha, \beta) = N/D$ defined in [](#equation:hamiltonian-expectation). This gives

$$
\label{equation:quotient-rule}
  \frac{\partial E(\alpha, \beta)}{\partial\gamma} = \frac{(\partial_\gamma N) D - N(\partial_\gamma D)}{D^2} = \frac{(\partial_\gamma N)}{D} E(\alpha, \beta) \frac{\partial_\gamma D}{D},
$$

where $\gamma = \alpha, \beta$. The numerator partial derivative is

$$
\begin{align*}
  \partial_\gamma N =& \frac{\partial}{\partial\gamma} \int_{\R^{3N}} \Psi(\mathbf{R}; \alpha, \beta) \hat{H}\Psi_T (\mathbf{R}; \alpha, \beta) \;\drm \mathbf{R} \\
  =& \int [(\partial_\gamma \Psi_T) \hat{H}\Psi_T + \Psi_T \hat{H} (\partial_\gamma \Psi_T)] \;\drm\mathbf{R}
\end{align*}
$$

Since $\hat{H}$ is Hermitian, we get

$$
  \int \Psi_T \hat{H} (\partial_\gamma \Psi_T) \;\drm\mathbf{R} = \int (\partial_\gamma \Psi_T) \hat{H} \Psi_T \;\drm\mathbf{R}
$$

Substituting the local energy $E_L$ defined in [](#equation:local-energy) results in

$$
  \partial_\gamma N = 2 \int (\partial_\gamma \Psi_T) \hat{H} \Psi_T \;\d\mathbf{R} = 2 \int |\Psi_T|^2 \left(\frac{\partial_\gamma \Psi_T}{\Psi_T} \right) E_L
$$

The denominator derivative is

$$
\begin{align*}
  \partial_\gamma D =& \frac{\partial}{\partial\gamma} \int_{\R^{3N}} |\Psi_T (\mathbf{R}; \alpha, \beta)|^2 \;\drm\mathbf{R} \\
  =& 2 \int \Psi_T (\partial_\gamma \Psi_T) \;\drm\mathbf{R} = 2 \int |\Psi_T|^2 \left(\frac{\partial_\gamma \Psi_T}{\Psi_T} \right)
\end{align*}
$$

Inserting into the quotient rule [](#equation:quotient-rule) yields

$$
\begin{align*}
  \frac{\partial E}{\partial\gamma} =& 2 \int \left(\frac{|\Psi_T|^2}{\int |\Psi_T|^2 \;\drm\mathbf{R}}\right) \left(\frac{\partial_\gamma \Psi_T}{\Psi_T} \right) E_L \;\drm\mathbf{R} \\
  &- 2 E(\alpha, \beta) \int \left(\frac{|\Psi_T|^2}{\int |\Psi_T|^2 \;\drm \mathbf{R}} \right) \left(\frac{\partial_\gamma \Psi_T}{\Psi_T} \right) \;\drm\mathbf{R}
\end{align*}
$$

Inserting the probability density $P_{\alpha,\beta}$ defined in [](#equation:wavefunction-pdf), we identify the expectation values

$$
  \Braket{\frac{\partial_\gamma \Psi_T}{\Psi_T} E_L} = \int P_{\alpha, \beta} (\mathbf{R}) \left(\frac{\partial_\gamma \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\mathbf{R}; \alpha, \beta)} \right) E_L (\mathbf{R}; \alpha, \beta) \;\drm\mathbf{R}
$$

and

$$
  \Braket{\frac{\partial_\gamma \Psi_T}{\Psi_T}} = \int P_{\alpha, \beta} (\mathbf{R}) \left(\frac{\partial_\gamma \Psi_T (\mathbf{R}; \alpha, \beta)}{\Psi_T (\mathbf{R}; \alpha, \beta)} \right) \;\drm\mathbf{R}
$$

Also, since $E(\alpha, \beta) = \braket{E_L}$ as defined in [](#equation:variation-energy-expectation), we finally arrive at

$$
  \frac{\partial E}{\partial\gamma} = 2\left(\Braket{\frac{\partial_\gamma \Psi_T}{\Psi_T} E_L} - \Braket{\frac{\partial_\gamma}{\Psi_T}} \braket{E_L} \right)
$$
