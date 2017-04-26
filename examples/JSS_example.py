from neurodesign import geneticalgorithm
EXP = geneticalgorithm.experiment(
 TR=1.2,
 n_trials=20,
 P = [0.3,0.3,0.4],
 C = [[1,-1,0],[0,1,-1]],
 n_stimuli = 3,
 rho = 0.3,
 stim_duration=1,
 ITImodel = "uniform",
 ITImin = 2,
 ITImax=4
 )

DES1 = geneticalgorithm.design(
  order = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1],
  ITI = [2]*20,
  experiment = EXP
)

DES1.designmatrix()

DES1.FCalc(weights=[0.25,0.25,0.25,0.25])

import matplotlib.pyplot as plt
plt.plot(DES1.Xconv)
plt.savefig("output/example_figure_1.pdf",format="pdf")

DES2 = geneticalgorithm.design(
    order = [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1],
    ITI = [2]*20,
    experiment=EXP
)
DES2.designmatrix(); DES2.FCalc(weights=[0.25,0.25,0.25,0.25])
print("Ff of Design 1: "+str(DES1.Ff))
print("Ff of Design 2: "+str(DES2.Ff))
print("Fd of Design 1: "+str(DES1.Fd))
print("Fd of Design 2: "+str(DES2.Fd))

DES3,DES4 = DES1.crossover(DES2,seed=2000)
print(DES3.order)
print(DES4.order)

POP = geneticalgorithm.population(
    experiment=EXP,
    weights=[0,0.5,0.25,0.25],
    preruncycles = 10000,
    cycles = 10000,
    folder = "./",
    seed=100
    )
POP.naturalselection()
