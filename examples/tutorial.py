# Adapted from https://lukas-snoek.com/NI-edu/fMRI-introduction/week_3/neurodesign.html

import os
# Neurodesigninternally paralellizes some computations using multithreading,
# which is a massive burden on the CPU. So let's limit the number of threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np 
import neurodesign
import matplotlib.pyplot as plt
from rich import print

TR = 1.6
rho = 0.6

n_stimuli = 2
stim_duration = 1
duration = 5*60
P = [0.5, 0.5]
t_pre = 0.1
t_post = 0.1

ITImodel = "uniform"
ITImin = 2
ITImax = 4

C = np.array([
    [1, -1],
    [-1, 1]
]) 

#  %%
exp = neurodesign.Experiment(
    TR=TR,
    rho=rho,
    n_stimuli=n_stimuli,
    stim_duration=stim_duration,
    P=P,
    duration=duration,
    t_pre=t_pre,
    t_post=t_post,
    ITImodel=ITImodel,
    ITImin=ITImin,
    ITImax=ITImax,
    C=C
)

#  %%
weights = [0, 1, 0, 0]  # order: Fe, Fd, Ff, Fc
outdes = 10

opt = neurodesign.Optimisation(
    experiment=exp,  # we have to give our previously created `exp` object to this class as well
    weights=weights,
    preruncycles=10,
    cycles=1,
    seed=2,
    outdes=outdes
)

opt.optimise()
opt.evaluate()

print(f"Onsets: {opt.bestdesign.onsets}")
print(f"Order: {opt.bestdesign.order}")

Xconv = opt.bestdesign.Xconv

plt.figure(figsize=(15, 5))
plt.plot(Xconv)
for ons, cond in zip(opt.bestdesign.onsets, opt.bestdesign.order):
    c = 'tab:blue' if cond == 0 else 'tab:orange'
    plt.plot([ons, ons], [0.35, 0.37], c=c, lw=2)
    
plt.legend(['Faces', 'Houses'])
plt.grid()
plt.xlim(0, Xconv.shape[0])
plt.show()