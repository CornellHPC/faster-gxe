# GENIE trace-estimator benchmark

`main.py` benchmarks randomized estimators for the covariate-adjusted trace block in GENIE's method-of-moments normal equations. It compares the original individual-space estimator with SNP-space estimators that share one genotype sketch across the additive and gene-by-environment components.

This is not a complete GENIE implementation: it does not simulate a phenotype, construct the full normal equations, fit variance components, compute heritability, or run a block jackknife.

## Run

The project uses JAX with CUDA 13 and Python 3.12 or newer.

```bash
uv sync
uv run python main.py
```

A larger single-precision speed benchmark:

```bash
uv run python main.py \
  --individuals 10000 \
  --snps 50000 \
  --environments 4 \
  --probes 10 \
  --repetitions 64 \
  --batch-size 8 \
  --probe-distribution rademacher \
  --dtype float32
```

Exact targets and theoretical variances are available for moderate problems:

```bash
uv run python main.py \
  --statistics \
  --individuals 1000 \
  --snps 2000 \
  --environments 3 \
  --probes 10 \
  --repetitions 1000 \
  --batch-size 32
```

Use `uv run python main.py --help` for all options and `uv run pytest` for the regression tests.

## Target

Let

- \(G\in\mathbb R^{N\times M}\) be the genotype matrix;
- \(e_i\in\mathbb R^N\) be component \(i\)'s environment vector and \(D_i=\operatorname{diag}(e_i)\);
- \(Q\in\mathbb R^{N\times r}\) have orthonormal columns spanning the fixed-effect covariates;
- \(P=I-QQ^\top\) project away from those covariates.

The first environment is the all-ones vector, representing the additive component. The other \(L-1\) environments represent GxE components. Define

\[
K_i=D_iGG^\top D_i,
\qquad
T_{ij}=\operatorname{tr}(K_iPK_jP).
\]

With projected features and kernels

\[
F_i=PD_iG,
\qquad
A_i=F_iF_i^\top=PK_iP,
\]

the symmetry and idempotence of \(P\) give

\[
T_{ij}=\operatorname{tr}(A_iA_j).
\]

The SNP-space matrix

\[
B_{ij}=F_i^\top F_j=G^\top D_iPD_jG
\]

provides the equivalent target

\[
T_{ij}=\operatorname{tr}(B_{ij}B_{ij}^\top)=\|B_{ij}\|_F^2.
\]

The benchmark reports this unnormalized quantity. For GENIE kernels normalized by \(1/M\), divide estimates and exact targets by \(M^2\), and variances by \(M^4\).

## Estimators

A probe \(z\) is isotropic when

\[
\mathbb E[z]=0,
\qquad
\mathbb E[zz^\top]=I.
\]

The supported independent-coordinate probes are standard Gaussian, with excess kurtosis \(\kappa=0\), and Rademacher, with entries in \(\{-1,+1\}\) and \(\kappa=-2\). Let \(q\) be the even probe count and \(h=q/2\).

### `genie`

For \(W=[w_1,\ldots,w_q]\in\mathbb R^{N\times q}\), compute \(Y_i=A_iW\) and

\[
\widehat T_{ij}^{\mathrm{genie}}
=\frac{1}{q}\langle Y_i,Y_j\rangle_F
=\frac{1}{q}\sum_{a=1}^q w_a^\top A_iA_jw_a.
\]

This is unbiased because \(\mathbb E[WW^\top/q]=I\). Each estimated component matrix is a symmetric positive-semidefinite Gram matrix. Its expensive step applies both \(G^\top\) and \(G\) separately for every component.

### Shared SNP-space sketch

Draw \(Z\in\mathbb R^{M\times q}\) once and form

\[
X_i=F_iZ=PD_iGZ,
\qquad
C_{ij}=X_i^\top X_j=Z^\top B_{ij}Z.
\]

All SNP-space estimators use

\[
\mathbb E[(u^\top B_{ij}v)^2]=\|B_{ij}\|_F^2
\]

for independent isotropic \(u,v\). The same \(GZ\) is reused for every component.

Split \(Z=[U,V]\) into two independent \(M\times h\) halves.

#### `match`

\[
\widehat T_{ij}^{\mathrm{match}}
=\frac{1}{h}\sum_{a=1}^h(u_a^\top B_{ij}v_a)^2.
\]

The \(h\) terms are independent.

#### `cross`

\[
\widehat T_{ij}^{\mathrm{cross}}
=\frac{1}{h^2}\sum_{a,b=1}^h(u_a^\top B_{ij}v_b)^2
=\frac{\|U^\top B_{ij}V\|_F^2}{h^2}.
\]

This averages `match` over every pairing. Its terms share probes, so its variance generally remains \(O(1/h)\), not \(O(1/h^2)\).

#### `leave`

\[
\widehat T_{ij}^{\mathrm{leave}}
=\frac{1}{q(q-1)}\sum_{a\ne b}(z_a^\top B_{ij}z_b)^2
=\frac{\|C_{ij}\|_F^2-\sum_a(C_{ij})_{aa}^2}{q(q-1)}.
\]

Every retained term uses independent columns. This is an order-two U-statistic and is valid for both supported probe distributions.

#### `rablk`

Gaussian probes are rotationally invariant. Rao-Blackwellizing `leave` over column rotations gives

\[
\widehat T_{ij}^{\mathrm{rablk}}
=\frac{(q+1)\|C_{ij}\|_F^2-\operatorname{tr}(C_{ij}^2)-\operatorname{tr}(C_{ij})^2}
{q(q-1)(q+2)}.
\]

Here \(\operatorname{tr}(C_{ij}^2)=\sum_{a,b}(C_{ij})_{ab}(C_{ij})_{ba}\). `leave` and `rablk` share the same full reduction, so reporting both adds negligible work.

For Gaussian probes,

\[
\operatorname{Var}(\mathrm{rablk})
\le \operatorname{Var}(\mathrm{leave})
\le \operatorname{Var}(\mathrm{cross})
\le \operatorname{Var}(\mathrm{match}).
\]

`genie`, `leave`, and `rablk` are symmetric for every draw. `match` and `cross` are unbiased for the symmetric target but can be asymmetric in a finite draw; downstream normal equations should symmetrize them.

## Exact variances

Statistics mode reports empirical and exact theoretical variances. For one component pair, write

\[
B=B_{ij},
\qquad
\theta=\|B\|_F^2,
\qquad
R=BB^\top,
\qquad
C=B^\top B,
\]

and define

\[
\alpha=\operatorname{tr}(R^2)=\operatorname{tr}(C^2),
\quad
\delta=\operatorname{tr}(RC),
\quad
\tau=\operatorname{tr}(B^2),
\quad
\gamma=\operatorname{tr}(B^4).
\]

The coordinate-dependent fourth-moment terms are

\[
a=\operatorname{diag}(R),
\quad
c=\operatorname{diag}(C),
\quad
b=\operatorname{diag}(B^2),
\]

\[
d_A=\|a\|_2^2,
\quad
d_C=\|c\|_2^2,
\quad
d_{AC}=a^\top c,
\quad
d_{B^2}=\|b\|_2^2,
\]

\[
e_4=\sum_{r,s}B_{rs}^4,
\qquad
e_{\mathrm{sym}}=\sum_{r,s}B_{rs}^2B_{sr}^2.
\]

### `genie`

For \(S_{ij}=(A_iA_j+A_jA_i)/2\),

\[
\operatorname{Var}(\widehat T_{ij}^{\mathrm{genie}})
=\frac{2\|S_{ij}\|_F^2+\kappa\|\operatorname{diag}(S_{ij})\|_2^2}{q}.
\]

### `match`

Let

\[
v_{\mathrm{pair}}
=2\theta^2+6\alpha+3\kappa(d_A+d_C)+\kappa^2e_4.
\]

Then

\[
\operatorname{Var}(\widehat T_{ij}^{\mathrm{match}})
=\frac{v_{\mathrm{pair}}}{h}.
\]

### `cross`

Let

\[
v_A=2\alpha+\kappa d_A,
\qquad
v_C=2\alpha+\kappa d_C.
\]

Then

\[
\operatorname{Var}(\widehat T_{ij}^{\mathrm{cross}})
=\frac{v_A+v_C}{h}
+\frac{v_{\mathrm{pair}}-v_A-v_C}{h^2}.
\]

### `leave`

Let

\[
\zeta_1=\alpha+\delta+\frac{\kappa}{4}(d_A+d_C+2d_{AC})
\]

and

\[
\zeta_2
=\theta^2+3\alpha+2\delta+\tau^2+\gamma
+\frac{3\kappa}{2}(d_A+d_C)+\kappa d_{AC}
+2\kappa d_{B^2}+\frac{\kappa^2}{2}(e_4+e_{\mathrm{sym}}).
\]

Then

\[
\operatorname{Var}(\widehat T_{ij}^{\mathrm{leave}})
=\frac{4(q-2)\zeta_1+2\zeta_2}{q(q-1)}.
\]

### `rablk`

For Gaussian probes, let \(H=(B+B^\top)/2\). Then

\[
\operatorname{Var}(\widehat T_{ij}^{\mathrm{rablk}})
=\operatorname{Var}(\widehat T_{ij}^{\mathrm{leave}})
-\frac{8\left[\operatorname{tr}(H^2)^2+2\operatorname{tr}(H^4)\right]}
{q(q-1)(q+2)}.
\]

The reduction is nonnegative and lower order in \(q\).

## JAX implementation

All numerical work runs on the device selected by JAX, including random data generation, EVD/SVD/QR, exact targets, theoretical variances, randomized estimators, and empirical summaries. Only finalized tables are transferred to the host for printing.

Synthetic inputs are constructed as follows:

1. Sample `clusters` prototype genotype rows from \(\{0,1,2\}^M\).
2. Copy prototypes across individuals and independently resample each copied SNP with probability \(0.3\).
3. Set the first environment to one and sample the remaining environments from a standard Gaussian.
4. Form a noisy low-rank population sketch, retain the leading `clusters` left singular vectors, and QR-orthonormalize them together with the environment vectors to obtain \(Q\).

The synthetic genotypes and environments are not standardized and are intended only to exercise the linear algebra.

### Modes and timing

- Speed mode runs only randomized estimators and is intended for large \(N,M\).
- Statistics mode additionally materializes projected features and individual-space kernels, then computes exact targets and variances. It is intended for moderate problems.
- Data generation and exact statistics are JIT-compiled; their setup times include compilation and execution.
- Estimator warm-up is reported separately from synchronized steady-state execution.
- Probe generation is included in estimator timings. Statistics collection and final table formatting are not.

Repetitions are evaluated in fixed-size batches. Environments and probes are vectorized inside each batch, and a short final batch is padded to avoid a second compilation. Timing is divided by the number of executed repetitions, including padding.

The five timed phases are `genie`, `feature_sketch`, `match_reduce`, `cross_reduce`, and `full_reduce`. Method totals count their dependencies once; `All timed execution phases` is the sum of running every phase, not the runtime of a single estimator.

### Complexity and memory

Let \(r\) be the covariate rank.

| Method | Per-repetition complexity |
|---|---:|
| `genie` | \(O(LNMq+LNrq+L^2Nq)\) |
| `match` | \(O(NMq+LNrq+L^2Nq)\) |
| `cross` | \(O(NMq+LNrq+L^2Nq^2/4)\) |
| `leave` | \(O(NMq+LNrq+L^2Nq^2)\) |
| `rablk` | \(O(NMq+LNrq+L^2Nq^2)\) |

The shared sketch changes the genotype multiplication from \(O(LNMq)\) to \(O(NMq)\), at the cost of an \(L^2\)-dependent reduction. `--batch-size` is the main throughput-versus-memory control.

Statistics mode additionally uses \(O(LNM)\) projected-feature storage and \(O(LN^2)\) kernel storage. Rademacher theory evaluates one \(M\times M\) cross-feature matrix at a time. Exact kernel construction costs \(O(LN^2M)\); Rademacher coordinate corrections can cost \(O(L^2NM^2)\).

## Options

| Option | Default | Meaning |
|---|---:|---|
| `--individuals` | `1000` | Individuals, \(N\) |
| `--snps` | `2000` | SNPs, \(M\) |
| `--clusters` | `4` | Prototype rows and retained population-PC directions |
| `--environments` | `3` | Components, including the additive component |
| `--probes` | `10` | Even total probe count, \(q\) |
| `--repetitions` | `64` | Independent probe repetitions |
| `--batch-size` | `8` | Repetitions per compiled call |
| `--oversample` | `10` | Extra population-sketch columns |
| `--seed` | `42` | Deterministic data seed; probes use the next seed |
| `--probe-distribution` | `gaussian` | `gaussian` or `rademacher` |
| `--dtype` | `float64` | Benchmark precision: `float32` or `float64` |
| `--statistics` | off | Compute exact targets, theory, and empirical summaries |

`--probes` must be even. In statistics mode, data and exact quantities use float64; estimator inputs are cast on-device when `--dtype float32` is requested.

Statistics tables report the exact target, empirical mean, relative bias, sample variance, theoretical variance, and their ratio. With enough repetitions, relative bias approaches zero and the empirical/theoretical ratio approaches one.

## Limitations

- Genotypes are dense; there is no Mailman algorithm or block-streaming reader.
- Only the additive/GxE trace block is benchmarked.
- One JAX device is used.
- Exact statistics do not scale to biobank-sized inputs.
- Theoretical formulas are implemented only for Gaussian and Rademacher probes.

## References

- Pazokitoroudi, A. et al. “A scalable and robust variance components method reveals insights into the architecture of gene-environment interactions underlying complex traits.” *The American Journal of Human Genetics* 111, 1462–1480 (2024). DOI: 10.1016/j.ajhg.2024.05.015.
- Hutchinson, M. “A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.” *Communications in Statistics—Simulation and Computation* 18, 1059–1076 (1989).
