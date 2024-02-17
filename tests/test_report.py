from neurodesign import optimisation, report


def test_report_smoke(exp, tmp_path):

    pop = optimisation(
        experiment=exp,
        weights=[0, 0.5, 0.25, 0.25],
        preruncycles=2,
        cycles=2,
        folder="./",
        seed=100,
    )
    pop.optimise()
    pop.download()

    report.make_report(pop, tmp_path / "test.pdf")
