from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

import neurodesign

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

EXP = neurodesign.experiment(
    TR=1.2,
    n_trials=20,
    P=[0.3, 0.3, 0.4],
    C=[[1, -1, 0], [0, 1, -1]],
    n_stimuli=3,
    rho=0.3,
    stim_duration=1,
    ITImodel="uniform",
    ITImin=2,
    ITImax=4,
)

DES1 = neurodesign.design(
    order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1], ITI=[2] * 20, experiment=EXP
)

DES1.designmatrix()

DES1.FCalc(weights=[0.25, 0.25, 0.25, 0.25])


plt.plot(DES1.Xconv)

plt.savefig(output_dir / "example_figure_1.pdf", format="pdf")

DES2 = neurodesign.design(
    order=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], ITI=[2] * 20, experiment=EXP
)
DES2.designmatrix()
DES2.FCalc(weights=[0.25, 0.25, 0.25, 0.25])
print(f"Ff of Design 1: {str(DES1.Ff)}")
print(f"Ff of Design 2: {str(DES2.Ff)}")
print(f"Fd of Design 1: {str(DES1.Fd)}")
print(f"Fd of Design 2: {str(DES2.Fd)}")

DES3, DES4 = DES1.crossover(DES2, seed=2000)
print(DES3.order)
print(DES4.order)


order = neurodesign.generate.order(
    nstim=4, ntrials=100, probabilities=[0.25, 0.25, 0.25, 0.25], ordertype="random", seed=1234
)
print(order[:10])
Counter(order)

iti, lam = neurodesign.generate.iti(
    ntrials=40, model="exponential", min=2, mean=3, max=8, resolution=0.1, seed=2134
)

print(iti[:10])
print(
    "mean ITI: %s \n\
      min ITI: %s \n\
      max ITI: %s"
    % (round(sum(iti) / len(iti), 2), round(min(iti), 2), round(max(iti), 2))
)

POP = neurodesign.optimisation(
    experiment=EXP, weights=[0, 0.5, 0.25, 0.25], preruncycles=10, cycles=100, folder="./", seed=100
)
POP.optimise()
