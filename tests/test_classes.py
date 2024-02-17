import pytest

from neurodesign import design, optimisation


def test_design_smoke(exp):

    des = design(
        order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )
    des.designmatrix()
    des.FCalc(weights=[0, 0.5, 0.25, 0.25])
    des.FdCalc()
    des.FcCalc()
    des.FfCalc()
    des.FeCalc()

    des.mutation(0.3, seed=2000)


def test_design_cross_over_smoke(exp):

    des = design(
        order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )

    des2 = design(
        order=[0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )

    des.crossover(des2, seed=2000)


@pytest.mark.parametrize("optimisation_type", ["GA", "simulation"])
def test_optimisation(exp, optimisation_type):

    pop = optimisation(
        experiment=exp,
        weights=[0, 0.5, 0.25, 0.25],
        preruncycles=2,
        cycles=2,
        folder="./",
        seed=100,
        optimisation=optimisation_type,
    )
    pop.optimise()
    pop.download()
    pop.evaluate()
