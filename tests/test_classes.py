import pytest

from neurodesign import Design, Optimisation


def test_design_smoke(exp):

    design = Design(
        order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )
    design.designmatrix()
    design.FCalc(weights=[0, 0.5, 0.25, 0.25])
    design.FdCalc()
    design.FcCalc()
    design.FfCalc()
    design.FeCalc()

    design.mutation(0.3, seed=2000)


def test_design_cross_over_smoke(exp):

    design = Design(
        order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )

    design_2 = Design(
        order=[0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1],
        ITI=[2] * 20,
        experiment=exp,
    )

    design.crossover(design_2, seed=2000)


@pytest.mark.parametrize("optimisation_type", ["GA", "simulation"])
def test_optimisation(exp, optimisation_type, tmp_path):

    population = Optimisation(
        experiment=exp,
        weights=[0, 0.5, 0.25, 0.25],
        preruncycles=2,
        cycles=2,
        folder=tmp_path,
        seed=100,
        optimisation=optimisation_type,
    )
    population.optimise()
    population.download()
    population.evaluate()
