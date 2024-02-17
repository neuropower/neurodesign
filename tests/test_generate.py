import pytest

from neurodesign import generate


@pytest.mark.parametrize("model", ["uniform", "exponential"])
def test_iti(model):

    generate.iti(ntrials=20, model=model, min=1, mean=2, max=4, seed=1234)

    generate.iti(ntrials=40, model=model, min=2, mean=3, max=8, resolution=0.1, seed=2134)


@pytest.mark.parametrize("ordertype", ["random", "blocked", "msequence"])
def test_order(ordertype):

    generate.order(
        nstim=4,
        ntrials=100,
        probabilities=[0.25, 0.25, 0.25, 0.25],
        ordertype=ordertype,
        seed=1234,
    )
