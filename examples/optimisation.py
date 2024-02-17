from pathlib import Path

from neurodesign import Experiment, Optimisation, report

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

exp = Experiment(
    TR=2,
    n_trials=100,
    P=[0.33, 0.33, 0.33],
    C=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [0, 1, -1]],
    n_stimuli=3,
    rho=0.3,
    resolution=0.1,
    stim_duration=1,
    t_pre=0,
    t_post=2,
    restnum=0,
    restdur=0,
    ITImodel="exponential",
    ITImin=1,
    ITImean=2,
    ITImax=4,
)

population = Optimisation(
    experiment=exp,
    weights=[0, 0.5, 0.25, 0.25],
    preruncycles=10,
    cycles=10,
    seed=1,
    outdes=5,
    folder=output_dir,
)

#########################
# run natural selection #
#########################

population.optimise()
population.download()
population.evaluate()

################
# step by step #
################

population.add_new_designs()
population.to_next_generation(seed=1)
population.to_next_generation(seed=1001)

#################
# export report #
#################
report.make_report(population, output_dir / "test.pdf")
