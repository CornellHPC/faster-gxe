import argparse
from collections import defaultdict
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

BASE_METHODS = ("genie", "match", "cross", "leave")


def project(array, covariates):
    return array - covariates @ (covariates.T @ array)


def sample_genotype(individuals, snps, clusters, key, dtype):
    prototype_key, mask_key, replacement_key = jax.random.split(key, 3)
    prototypes = jax.random.randint(prototype_key, (clusters, snps), 0, 3, dtype=jnp.int8)
    copied = prototypes[jnp.arange(clusters, individuals) % clusters]
    mask = jax.random.bernoulli(mask_key, 0.3, copied.shape)
    replacements = jax.random.randint(replacement_key, copied.shape, 0, 3, dtype=jnp.int8)
    return jnp.concatenate((prototypes, jnp.where(mask, replacements, copied))).astype(dtype)


@partial(jax.jit, static_argnames=("individuals", "snps", "clusters", "environment_count", "oversample", "dtype"))
def generate_data(individuals, snps, clusters, environment_count, oversample, seed, dtype):
    genotype_key, environment_key, omega_key, noise_key = jax.random.split(jax.random.key(seed), 4)
    genotype = sample_genotype(individuals, snps, clusters, genotype_key, dtype)
    environments = jax.random.normal(environment_key, (environment_count, individuals), dtype).at[0].set(1.0)

    width = clusters + oversample
    omega = jax.random.normal(omega_key, (snps, width), dtype)
    covariance = omega.T @ omega
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    covariance_root = (eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    noise = jax.random.normal(noise_key, (individuals, width), dtype)
    population_sketch = genotype @ omega + 0.1 * noise @ covariance_root
    population_vectors, _, _ = jnp.linalg.svd(population_sketch, full_matrices=False)
    covariates, _ = jnp.linalg.qr(jnp.concatenate((population_vectors[:, :clusters], environments.T), axis=1), mode="reduced")
    return genotype, environments, covariates


@partial(jax.jit, static_argnames=("probes", "distribution", "methods"))
def compute_statistics(genotype, environments, covariates, probes, distribution, methods):
    features = project(environments[:, :, None] * genotype, covariates)
    kernels = features @ jnp.swapaxes(features, -1, -2)
    component_count = features.shape[0]
    half = probes // 2
    kappa = {"gaussian": 0.0, "rademacher": -2.0}[distribution]
    rows, columns = jnp.tril_indices(component_count)

    def pair_statistics(pair):
        i, j = pair
        left, right = features[i], features[j]
        kernel_left, kernel_right = kernels[i], kernels[j]
        kernel_product = kernel_left @ kernel_right
        symmetric = 0.5 * (kernel_product + kernel_product.T)
        theta = jnp.vdot(kernel_left, kernel_right)
        alpha = jnp.einsum("ij,ji->", kernel_product, kernel_product)
        variance_genie = (2.0 * jnp.vdot(symmetric, symmetric) + kappa * jnp.vdot(jnp.diag(symmetric), jnp.diag(symmetric))) / probes

        def diagonal(_):
            return alpha, alpha, theta, theta, alpha

        def off_diagonal(_):
            cross_kernel = left @ right.T
            cross_kernel_square = cross_kernel @ cross_kernel
            tau = jnp.einsum("ij,ji->", cross_kernel, cross_kernel)
            gamma = jnp.einsum("ij,ji->", cross_kernel_square, cross_kernel_square)
            delta = jnp.vdot(kernel_right @ cross_kernel, cross_kernel @ kernel_left)
            beta = jnp.einsum("ij,ji->", kernel_product, cross_kernel_square)
            symmetric_square = 0.5 * (theta + tau)
            symmetric_fourth = (gamma + 4.0 * beta + 2.0 * delta + alpha) / 8.0
            return delta, gamma, tau, symmetric_square, symmetric_fourth

        delta, gamma, tau, symmetric_square, symmetric_fourth = jax.lax.cond(i == j, diagonal, off_diagonal, operand=None)

        dA = dC = dAC = dB2 = e4 = e_sym = jnp.zeros((), genotype.dtype)
        if distribution == "rademacher":
            matrix = left.T @ right
            squared = matrix * matrix
            row_norms = squared.sum(axis=1)
            column_norms = squared.sum(axis=0)
            paired_products = matrix * matrix.T
            diagonal_b_squared = paired_products.sum(axis=1)
            dA = jnp.vdot(row_norms, row_norms)
            dC = jnp.vdot(column_norms, column_norms)
            dAC = jnp.vdot(row_norms, column_norms)
            dB2 = jnp.vdot(diagonal_b_squared, diagonal_b_squared)
            e4 = jnp.sum(squared * squared)
            e_sym = jnp.sum(paired_products * paired_products)

        variance_pair = 2.0 * theta**2 + 6.0 * alpha + 3.0 * kappa * (dA + dC) + kappa**2 * e4
        variance_left = 2.0 * alpha + kappa * dA
        variance_right = 2.0 * alpha + kappa * dC
        variance_cross = (variance_left + variance_right) / half + (variance_pair - variance_left - variance_right) / half**2
        zeta1 = alpha + delta + 0.25 * kappa * (dA + dC + 2.0 * dAC)
        zeta2 = theta**2 + 3.0 * alpha + 2.0 * delta + tau**2 + gamma + 1.5 * kappa * (dA + dC) + kappa * dAC + 2.0 * kappa * dB2 + 0.5 * kappa**2 * (e4 + e_sym)
        variances = {"genie": variance_genie, "match": variance_pair / half, "cross": variance_cross, "leave": (4.0 * (probes - 2) * zeta1 + 2.0 * zeta2) / (probes * (probes - 1))}
        if distribution == "gaussian":
            variances["rablk"] = variances["leave"] - 8.0 * (symmetric_square**2 + 2.0 * symmetric_fourth) / (probes * (probes - 1) * (probes + 2))
        return theta, jnp.maximum(jnp.stack([variances[method] for method in methods]), 0.0)

    targets, variances = jax.lax.map(pair_statistics, (rows, columns))
    truth = jnp.zeros((component_count, component_count), genotype.dtype)
    truth = truth.at[rows, columns].set(targets).at[columns, rows].set(targets)
    theory = {}
    for index, method in enumerate(methods):
        matrix = jnp.zeros_like(truth)
        values = variances[:, index]
        theory[method] = matrix.at[rows, columns].set(values).at[columns, rows].set(values)
    return truth, theory


def build_kernels(distribution, dtype, probes):
    half = probes // 2

    def draw_batch(keys, rows, tag):
        if distribution == "gaussian":
            return jax.vmap(lambda key: jax.random.normal(jax.random.fold_in(key, tag), (rows, probes), dtype))(keys)
        return jax.vmap(lambda key: jax.random.rademacher(jax.random.fold_in(key, tag), (rows, probes)).astype(dtype))(keys)

    def genie(keys, genotype, environments, covariates):
        random_vectors = draw_batch(keys, genotype.shape[0], 0)
        values = environments[None, :, :, None] * project(random_vectors, covariates)[:, None]
        values = genotype @ (genotype.T @ values)
        sketches = project(environments[None, :, :, None] * values, covariates)
        return jnp.einsum("blnq,bjnq->bljq", sketches, sketches).mean(axis=-1)

    def feature_sketch(keys, genotype, environments, covariates):
        random_vectors = draw_batch(keys, genotype.shape[1], 1)
        genotype_sketch = genotype @ random_vectors
        return project(environments[None, :, :, None] * genotype_sketch[:, None], covariates)

    def match(sketches):
        left, right = sketches[..., :half], sketches[..., half:]
        products = jnp.einsum("blnh,bjnh->bljh", left, right)
        return jnp.mean(products * products, axis=-1)

    def cross(sketches):
        left, right = sketches[..., :half], sketches[..., half:]
        grams = jnp.swapaxes(left, -1, -2)[:, :, None] @ right[:, None]
        return jnp.mean(grams * grams, axis=(-2, -1))

    def full_reduce(sketches):
        grams = jnp.swapaxes(sketches, -1, -2)[:, :, None] @ sketches[:, None]
        diagonal = jnp.diagonal(grams, axis1=-2, axis2=-1)
        frobenius_square = jnp.sum(grams * grams, axis=(-2, -1))
        leave = (frobenius_square - jnp.sum(diagonal * diagonal, axis=-1)) / (probes * (probes - 1))
        if distribution != "gaussian":
            return (leave,)
        trace_of_square = jnp.sum(grams * jnp.swapaxes(grams, -1, -2), axis=(-2, -1))
        trace = jnp.sum(diagonal, axis=-1)
        rablk = ((probes + 1) * frobenius_square - trace_of_square - trace**2) / (probes * (probes - 1) * (probes + 2))
        return leave, rablk

    return {"genie": jax.jit(genie), "feature_sketch": jax.jit(feature_sketch), "match_reduce": jax.jit(match), "cross_reduce": jax.jit(cross), "full_reduce": jax.jit(full_reduce)}


def timed(times, name, function, *args):
    start = perf_counter()
    result = function(*args)
    jax.block_until_ready(result)
    times[name] += perf_counter() - start
    return result


def run_batch(kernels, keys, genotype, environments, covariates, times, distribution):
    estimates = {"genie": timed(times, "genie", kernels["genie"], keys, genotype, environments, covariates)}
    sketches = timed(times, "feature_sketch", kernels["feature_sketch"], keys, genotype, environments, covariates)
    estimates["match"] = timed(times, "match_reduce", kernels["match_reduce"], sketches)
    estimates["cross"] = timed(times, "cross_reduce", kernels["cross_reduce"], sketches)
    full = timed(times, "full_reduce", kernels["full_reduce"], sketches)
    estimates["leave"] = full[0]
    if distribution == "gaussian":
        estimates["rablk"] = full[1]
    return estimates


def run_benchmark(genotype, environments, covariates, probes, repetitions, batch_size, seed, distribution, methods, collect_samples):
    batch_size = min(batch_size, repetitions)
    executed_repetitions = ((repetitions + batch_size - 1) // batch_size) * batch_size
    keys = jax.random.split(jax.random.key(seed), executed_repetitions)
    kernels = build_kernels(distribution, genotype.dtype, probes)

    warmup = defaultdict(float)
    run_batch(kernels, keys[:batch_size], genotype, environments, covariates, warmup, distribution)

    samples = {method: [] for method in methods} if collect_samples else None
    execution = defaultdict(float)
    for start in range(0, repetitions, batch_size):
        count = min(batch_size, repetitions - start)
        estimates = run_batch(kernels, keys[start : start + batch_size], genotype, environments, covariates, execution, distribution)
        if samples is not None:
            for method, values in estimates.items():
                samples[method].append(values[:count])

    if samples is not None:
        samples = {method: jnp.concatenate(batches) for method, batches in samples.items()}
        jax.block_until_ready(samples)

    return samples, warmup, execution, executed_repetitions, batch_size


def print_statistics(samples, truth, theory):
    methods = tuple(samples)
    summaries = {}
    for method, values in samples.items():
        mean = jnp.mean(values, axis=0)
        empirical = jnp.var(values, axis=0, ddof=1) if len(values) > 1 else jnp.full_like(truth, jnp.nan)
        theoretical = theory[method]
        relative_bias = jnp.where(truth != 0, (mean - truth) / truth, jnp.nan)
        ratio = jnp.where(theoretical > 0, empirical / theoretical, jnp.nan)
        summaries[method] = mean, relative_bias, empirical, theoretical, ratio
    truth, summaries = jax.device_get((truth, summaries))

    print("\nEstimator statistics")
    for i in range(truth.shape[0]):
        for j in range(i + 1):
            target = truth[i, j]
            print(f"\npair=({i}, {j}) truth={target:.8e}")
            print(f"{'method':<8} {'mean':>14} {'rel_bias':>12} {'emp_var':>14} {'theory_var':>14} {'emp/theory':>12}")
            for method in methods:
                summary = summaries[method]
                mean, relative_bias, empirical, theoretical, ratio = (value[i, j] for value in summary)
                print(f"{method:<8} {mean:14.6e} {relative_bias:12.4e} {empirical:14.6e} {theoretical:14.6e} {ratio:12.4f}")


def print_timings(setup, warmup, execution, repetitions, executed_repetitions, distribution):
    dependencies = {"genie": ("genie",), "match": ("feature_sketch", "match_reduce"), "cross": ("feature_sketch", "cross_reduce"), "leave": ("feature_sketch", "full_reduce")}
    complexity = {"genie": "O(LNMq + LNrq + L^2Nq)", "match": "O(NMq + LNrq + L^2Nq)", "cross": "O(NMq + LNrq + L^2Nq^2/4)", "leave": "O(NMq + LNrq + L^2Nq^2)"}
    if distribution == "gaussian":
        dependencies.update({"rablk": ("feature_sketch", "full_reduce"), "leave+rablk": ("feature_sketch", "full_reduce")})
        complexity.update({"rablk": complexity["leave"], "leave+rablk": complexity["leave"]})

    print("\nOne-time setup")
    for name, seconds in setup.items():
        print(f"{name:<22} {seconds:12.6f}s")

    print("\nJAX warm-up: compilation plus one batch")
    for name, seconds in warmup.items():
        print(f"{name:<22} {seconds:12.6f}s")
    print(f"{'total':<22} {sum(warmup.values()):12.6f}s")

    print(f"\nExecution over {repetitions} requested repetitions ({executed_repetitions} executed with padding)")
    print(f"{'phase':<22} {'total_s':>12} {'per_executed_rep_s':>20}")
    for name, seconds in execution.items():
        print(f"{name:<22} {seconds:12.6f} {seconds / executed_repetitions:20.6f}")

    print("\nMethod totals from dependencies")
    print(f"{'method':<14} {'total_s':>12} {'per_executed_rep_s':>20}  complexity")
    for method, phases in dependencies.items():
        total = sum(execution[phase] for phase in phases)
        print(f"{method:<14} {total:12.6f} {total / executed_repetitions:20.6f}  {complexity[method]}")
        print(f"  depends on: {' + '.join(phases)}")

    print(f"\nAll timed execution phases: {sum(execution.values()):.6f}s")
    print("N=individuals, M=SNPs, L=components, q=probes, r=covariate rank")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--individuals", type=int, default=1000)
    parser.add_argument("--snps", type=int, default=2000)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--environments", type=int, default=3)
    parser.add_argument("--probes", type=int, default=10, help="even total probe count")
    parser.add_argument("--repetitions", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8, help="repetitions evaluated together by each JAX call")
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-distribution", choices=("gaussian", "rademacher"), default="gaussian")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--statistics", action="store_true", help="compute exact traces and theoretical variances; otherwise benchmark speed only")
    args = parser.parse_args()

    if min(args.individuals, args.snps, args.clusters, args.environments, args.repetitions, args.batch_size) < 1:
        parser.error("dimensions, repetitions, and batch size must be positive")
    if args.clusters > min(args.individuals, args.snps):
        parser.error("clusters cannot exceed min(individuals, snps)")
    if args.clusters + args.environments >= args.individuals:
        parser.error("clusters + environments must be smaller than individuals")
    if args.probes < 2 or args.probes % 2:
        parser.error("probes must be an even integer of at least 2")
    if args.oversample < 0:
        parser.error("oversample must be nonnegative")
    return args


def main():
    args = parse_args()
    methods = BASE_METHODS + (("rablk",) if args.probe_distribution == "gaussian" else ())
    benchmark_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    data_dtype = jnp.float64 if args.statistics else benchmark_dtype
    setup = {}

    start = perf_counter()
    genotype, environments, covariates = generate_data(args.individuals, args.snps, args.clusters, args.environments, args.oversample, args.seed, data_dtype)
    jax.block_until_ready((genotype, environments, covariates))
    setup["data"] = perf_counter() - start

    truth = theory = None
    if args.statistics:
        start = perf_counter()
        truth, theory = compute_statistics(genotype, environments, covariates, args.probes, args.probe_distribution, methods)
        jax.block_until_ready((truth, theory))
        setup["ground_truth+theory"] = perf_counter() - start

    if data_dtype != benchmark_dtype:
        start = perf_counter()
        genotype = genotype.astype(benchmark_dtype)
        environments = environments.astype(benchmark_dtype)
        covariates = covariates.astype(benchmark_dtype)
        jax.block_until_ready((genotype, environments, covariates))
        setup["benchmark_cast"] = perf_counter() - start

    samples, warmup, execution, executed_repetitions, batch_size = run_benchmark(genotype, environments, covariates, args.probes, args.repetitions, args.batch_size, args.seed + 1, args.probe_distribution, methods, args.statistics)

    print(f"individuals={args.individuals} snps={args.snps} clusters={args.clusters} environments={args.environments} probes={args.probes} repetitions={args.repetitions} batch_size={batch_size} distribution={args.probe_distribution} methods={','.join(methods)} dtype={args.dtype} statistics={args.statistics} seed={args.seed} covariate_rank={covariates.shape[1]} backend={jax.default_backend()} device={jax.devices()[0]}")
    if args.probe_distribution == "gaussian":
        print("rablk added because Gaussian probes admit the Haar Rao-Blackwell estimator.")
    if args.statistics:
        print_statistics(samples, truth, theory)
    print_timings(setup, warmup, execution, args.repetitions, executed_repetitions, args.probe_distribution)


if __name__ == "__main__":
    main()
