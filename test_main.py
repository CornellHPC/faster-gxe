import jax
import jax.numpy as jnp
import pytest

from main import BASE_METHODS, compute_statistics, generate_data, project


DATA_ARGS = (24, 32, 2, 3, 2)


def make_data(seed=7):
    return generate_data(*DATA_ARGS, seed, jnp.float64)


def test_generate_data_is_deterministic_and_well_formed():
    first = make_data()
    repeated = make_data()
    changed = make_data(seed=8)
    genotype, environments, covariates = first

    assert all(array.device.platform == jax.default_backend() for array in first)
    assert genotype.shape == DATA_ARGS[:2]
    assert environments.shape == (DATA_ARGS[3], DATA_ARGS[0])
    assert covariates.shape == (DATA_ARGS[0], DATA_ARGS[2] + DATA_ARGS[3])
    assert all(array.dtype == jnp.float64 for array in first)
    assert all(bool(jnp.array_equal(left, right)) for left, right in zip(first, repeated))
    assert not bool(jnp.array_equal(genotype, changed[0]))
    assert bool(jnp.all((genotype >= 0) & (genotype <= 2) & (genotype == jnp.round(genotype))))
    assert bool(jnp.all(environments[0] == 1))
    assert bool(jnp.allclose(covariates.T @ covariates, jnp.eye(covariates.shape[1]), rtol=1e-11, atol=1e-11))


@pytest.mark.parametrize("distribution", ("gaussian", "rademacher"))
def test_exact_statistics_match_direct_target_and_theory_invariants(distribution):
    genotype, environments, covariates = make_data()
    methods = BASE_METHODS + (("rablk",) if distribution == "gaussian" else ())
    truth, theory = compute_statistics(genotype, environments, covariates, 10, distribution, methods)

    features = project(environments[:, :, None] * genotype, covariates)
    kernels = features @ jnp.swapaxes(features, -1, -2)
    direct = jnp.einsum("iab,jab->ij", kernels, kernels)

    assert truth.device.platform == jax.default_backend()
    assert bool(jnp.allclose(truth, direct, rtol=1e-11, atol=1e-8))
    assert bool(jnp.allclose(truth, truth.T, rtol=0, atol=0))
    assert set(theory) == set(methods)
    for variance in theory.values():
        assert variance.device.platform == jax.default_backend()
        assert bool(jnp.allclose(variance, variance.T, rtol=0, atol=0))
        assert bool(jnp.all(variance >= 0))

    assert bool(jnp.all(theory["cross"] <= theory["match"] * (1 + 1e-12) + 1e-8))
    assert bool(jnp.all(theory["leave"] <= theory["cross"] * (1 + 1e-12) + 1e-8))
    if distribution == "gaussian":
        assert bool(jnp.all(theory["rablk"] <= theory["leave"] * (1 + 1e-12) + 1e-8))
