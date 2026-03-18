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
    \frac{1}{2} m\omega_\text{ho}^2 (x^2 + y^2) + \omega_z^2 z^2,\quad (E)
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

For convenience, we consider the isotropic case $\beta 1$ deriving an analytical expression for the local energy $E_L$ in a $d$-dimensional system. In this case, the trial wave function [](#equation:trial-wavefunction-noninteracting) factorizes as

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
  \frac{\nabla_i^2 \phi(\mathbf{r}_i)}{\phi(\mathbf{r}_i)} = 4\alpha^2 r_i^2 - 2d\alpha
$$

Assuming a spherical trap potential in [](#equation:trap-potential), the local energy [](#equation:local-energy-noninteracting) becomes

$$
  E_L (\mathbf{R}) = \frac{d\hbar^2 \alpha N}{m} + \sum_{i=1}^N \left(\frac{1}{2} m\omega_\text{ho}^2 - \frac{2\hbar^2 \alpha^2}{m} \right) r_i^2.
$$

We note that the sum vanishes when

$$
  \alpha = \frac{m\omega_\text{ho}}{2\hbar},
$$

corresponding to the exact harmonic-oscillator ground state. In this case, the local energy becomes the constant

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

where $\phi(\mathbf{r}_i) = g(\alpha,\beta,\mathbf{r}_i)$, $r_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$ and $f(r_{ij}) = e^{u(r_{ij})}$. In this form, the Laplacian of $\Psi_T$ can be shown to satisfy the identity

$$
\label{equation:trial-wavefunction-interacting-laplacian}
\begin{equation*}
\begin{split}
  \frac{1}{\Psi_T} \nabla_k^2 \Psi_T (\mathbf{r}) =& \frac{\nabla_k^2 \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + 2\frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} \left(\sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_j)}{r_{kj}} u'(r_{kj}) \right) \\
  &+ \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj}} u'(r_{ki}) u'(r_{kj}) \\
  &+ \sum_{j\neq k} \left(u'' (r_{kj}) + \frac{2}{r_{kj}} u' (r_{kj}) \right)
\end{split}
\end{equation*}
$$

## Metropolis Algorithm

The ground state energy of the bose gas correlated model can be estimated using variational Monte Carlo methods. The expectation value of the Hamiltonian $\hat{H}$, in the state $\Psi_T (\mathbf{R}, \alpha, \beta)$ is given by

$$
\begin{equation*}
\label{equation:hamiltonian-expectation}
\begin{split}
  E(\alpha, \beta) :=& \braket{H}_{\Psi_T} = \frac{\braket{\Psi_T (\mathbf{R}, \alpha, \beta), \hat{H} \Psi_T (\mathbf{R}, \alpha, \beta)}}{\braket{\Psi(\mathbf{R}, \alpha, \beta), \Psi_T (\mathbf{R}, \alpha, \beta)}} \\
  =& \frac{\int_{\R^{3N}} \Psi_T^* (\mathbf{R}, \alpha, \beta) \hat{H}\Psi_T (\mathbf{R}, \alpha, \beta) \;\d\mathbf{R}}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}, \alpha, \beta)|^2 \;\d\mathbf{R}}
\end{split}
\end{equation*}
$$

By the Rayleigh-Ritz principle $E(\alpha, \beta) \geq E_0$, where $E_0$ is the ground-state energy. Thus, we can approximate $E_0$ by minimizing $E(\alpha, \beta)$ over the parameters $(\alpha, \beta)$.

To enable Monte Carlo estimation, we expand the variational energy [](#equation:trial-wavefunction) in terms of the probability density function

$$
  P_{\alpha,\beta} (\mathbf{R}) = \frac{|\Psi_T (\mathbf{R}, \alpha, \beta)|^2}{\int_{\R^{3N}} |\Psi_T (\mathbf{R}, \alpha, \beta)|^2 \;\d\mathbf{R}}
$$

Substituting the local energy [](#equation:trial-wavefunction), the variational energy can be written as the expectation value

$$
\label{equation:variation-energy-expectation}
  E(\alpha, \beta) = \int_{\R^{3N}} P_{\alpha,\beta} (\mathbf{R}) E_L (\mathbf{R}, \alpha, \beta) \;\d\mathbf{R}
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
  A(\mathbf{R},\mathbf{R}') = \min\Set{1, \frac{|\Psi_T (\mathbf{R}', \alpha,\beta)|^2}{|\Psi_T (\mathbf{R}, \alpha, \beta)|^2}}
$$

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

## Parameter Optimization

# Results

## Non-Interacting Case

The numerical approach using finite difference derivation to evaluate the Laplacian of $\Psi_T$ requires $O(2*n*d)$ operations, since it requires calculating the wavefunction twice per coordinate, while the analytic expression scales as $O(1)$.

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
  V_\text{ext} (\mathbf{r}_i) = \frac{1}{2}m\omega^2_\text{ho} r_i^2
$$

into [](#equation:local-energy-spherical-noninteracting), we find

$$
\begin{align*}
  E_L (\mathbf{R}) =& \sum_{i=1}^N \left(-\frac{\hbar^2}{2m} (4\alpha^2 r_i^2 - 2d\alpha) + \frac{1}{2} m\omega_\text{ho}^2 r_i^2 \right) \\
  =& \sum_{i=1}^N \left(\frac{d\hbar^2 \alpha}{m} - \frac{2\hbar^2 \alpha^2}{m} r_i^2 + \frac{1}{2} m\omega_\text{ho}^2 r_i^2 \right) \\
  =& \frac{d\hbar^2 \alpha N}{m} + \sum_{i=1}^N \left(\frac{1}{2}m\omega_\text{ho}^2 - \frac{2\hbar^2 \alpha^2}{m} \right) r_i^2
\end{align*}
$$

## Local Energy in the Interacting Case

To derive the Laplacian identity [](#equation:trial-wavefunction-interacting-laplacian), we introduce

$$
  A(\mathbf{R}) = \prod_{i=1}^N \phi(\mathbf{r}_i),\quad B(\mathbf{R}) = \exp\left(\sum_{j < k} u(r_{jk}) \right), 
$$

such that $\Psi_T$ in [](#equation:trial-wavefunction-interacting) can be factored as

$$
  \Psi_T (\mathbf{R}) = A(\mathbf{R}) B(\mathbf{R})
$$

Taking the gradient and appling the product rule yields

$$
  \nabla_k \Psi_T  = (\nabla_k A) B + A \nabla_k B
$$

The first gradient $\nabla_k A$ is given by

$$
  \nabla_k A(\mathbf{R}) = \nabla_k \left(\prod_{i=1}^N \phi(\mathbf{r}_i) \right) = \left(\prod_{i\neq k} \phi(\mathbf{r}_i) \right) \nabla_k \phi(\mathbf{r}_k)
$$

since only the factor $\phi(\mathbf{r}_k)$ depends on $\mathbf{r}_k$. To calculate the second gradient $\nabla_k B$, we first substitute $U(\mathbf{R}) = \sum_{j < k} u(r_{ij})$, such that $B = e^U$. The chain rule yields $\nabla_k B = B \nabla_k U$, hence

$$
  \nabla_k B(\mathbf{R}) = B(\mathbf{R}) \sum_{j\leq k} \nabla_k u(r_{jk})
$$

Combining the terms leads to

$$
\begin{align*}
  \nabla_k \Psi_T (\mathbf{R}) =& \exp\left(\sum_{j < m} u(r_{jm}) \right) \left(\prod_{i\neq k} \phi(\mathbf{r}_i) \right) \nabla_k \phi(\mathbf{r}_k) \\
  &+ \left(\prod_{i=1}^N \phi(\mathbf{r}_i) \right) \exp\left(\sum_{j < m} u(r_{jm}) \right) \sum_{l\neq k} \nabla_k u(r_{kl}) \\
  =& \Psi_T (\mathbf{R}) 
\end{align*}
$$

Factoring out $\Psi_T$, we arrive at

$$
  \frac{\nabla_k \Psi_T (\mathbf{R})}{\Psi_T (\mathbf{R})} = \frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + \sum_{l\neq k} \nabla_k u(r_{kl})
$$

To evaluate $\nabla_k u(r_{kj})$, we apply the chain rule, giving $\nabla_k u(r_{kj}) = u' (r_{kj}) \nabla_k r_kj$. Introducing $\mathbf{r} := \mathbf{r}_k - \mathbf{r}_j$, such that $r_{kj} = |\mathbf{r}|$ we get

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

To derive the Laplacian of $\Psi_T$, we define

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
  |\mathbf{B}_k|^2 = \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj}} u'(r_{ki}) u'(r_{kj})
$$

The divergence term in [](#equation:laplacian-identity-simplified) is given by the product rule

$$
  \nabla_k \cdot \mathbf{F}_k = \nabla_k \cdot \mathbf{A}_k + \nabla_k \cdot \mathbf{B}_k,
$$

where the first term evaluates to

$$
\begin{align*}
  \nabla_k \cdot \mathbf{A}_k =& \nabla_k \cdot \left(\frac{\nabla_k \phi}{\phi} \right) = \nabla_k \left(\frac{1}{\phi} \right) \cdot \nabla_k \phi + \frac{1}{\phi} \nabla_k^2 \phi \\
  =& -\frac{1}{\phi^2} (\nabla_k \phi) \cdot (\nabla_k \phi) - \frac{1}{\phi} \nabla_k^2 \phi  = \frac{\nabla_k^2 \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} - \left|\frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} \right|^2
\end{align*}
$$

Defining

$$
  \mathbf{x} := \mathbf{r}_k - \mathbf{r}_j,\quad x := r_{kj},\quad v(x) = \frac{u'(x)}{x}
$$

we can write

$$
  u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}} = v(x) \mathbf{x}
$$

Using $\nabla_\mathbf{x} = \nabla_r$, the divergence product rule yields

$$
\begin{align*}
  \nabla_\mathbf{x} \cdot (v(x) \mathbf{x}) =& (\nabla_\mathbf{x} v) \cdot \mathbf{x} + v(x) \underbrace{\nabla_\mathbf{x} \cdot \mathbf{x}}_{=3} = v'(x) \underbrace{(\nabla_\mathbf{x} x)}_{=\mathbf{x}/x} \cdot \mathbf{x} + 3v(x) \\
  =& x v'(x) + 3v(x)
\end{align*} 
$$

Since

$$
  g'(r) = \frac{\mathrm{d}}{\mathrm{d}r} \left(\frac{u'(r)}{r} \right) = \frac{u''(r)r - u'(r)}{r^2}
$$

we get

$$
  \nabla_\mathbf{x} \cdot (v(x) \mathbf{x}) = \frac{u''(x) - u'(x)}{x} + \frac{3u'(x)}{x} = u''(x) + \frac{2}{x} u'(x)
$$

Substituting back, we get

$$
\begin{align*}
  \nabla_k \cdot \left(u'(r_{kj}) \frac{\mathbf{r}_k - \mathbf{r}_j}{r_{kj}} \right) = u'' (r_{kj}) + \frac{2}{r_{kj}} u'(r_{kj})
\end{align*} 
$$

and

$$
  \nabla_k \cdot \mathbf{B}_k = \sum_{j\neq k} \left(u'' (r_{kj}) + \frac{2}{r_{kj}} u'(r_{kj}) \right)
$$

Combining the terms, we end up with

$$
\begin{align*}
  \frac{1}{\Psi_T} \nabla_k^2 \Psi_T (\mathbf{r}) =& \frac{\nabla_k^2 \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} + 2\frac{\nabla_k \phi(\mathbf{r}_k)}{\phi(\mathbf{r}_k)} \left(\sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_j)}{r_{kj}} u'(r_{kj}) \right) \\
  &+ \sum_{i\neq k} \sum_{j\neq k} \frac{(\mathbf{r}_k - \mathbf{r}_i)(\mathbf{r}_k - \mathbf{r}_j)}{r_{ki} r_{kj}} u'(r_{ki}) u'(r_{kj}) \\
  &+ \sum_{j\neq k} \left(u'' (r_{kj}) + \frac{2}{r_{kj}} u' (r_{kj}) \right)
\end{align*}
$$
