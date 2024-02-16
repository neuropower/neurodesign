# # Neurodesign comparison of design generators
#
# In this notebook, we will compare 3 methods to generate an experimental design:
# - a design optimised using the genetic algorithm
# - a design optimised using simulations
# - a randomly drawn design
#
# We will do so using simulations: what is the resulting observed power when we simulate experiments according to the three designs.

# In[1]:

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t

from neurodesign import experiment, optimisation

cycles = 1000
sims = 10000


# ## Optimise designs

# First we define the experiment.  We will optimise an experiment with a TR of 2 seconds and 250 trials of 0.5 seconds each.  There are 4 stimulus types, and we are interested in the shared effect of the first and second stimulus versus baseline, as well as the difference between the first and the fourth stimulus.  We assume an autoregressive temporal autocorrelation of 0.3.
#
# We sample ITI's from a truncated exponential distribution with minimum 0.3 seconds and maximum 4 seconds, and the mean is 1 second.

# In[2]:

# define the experiment
EXP = experiment(
    TR=2,
    n_trials=450,
    P=[0.25, 0.25, 0.25],
    C=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, -1]],
    n_stimuli=3,
    rho=0.3,
    resolution=0.1,
    stim_duration=1,
    ITImodel="exponential",
    ITImin=0.3,
    ITImean=1,
    ITImax=4,
)


# In[3]:

POP_Max = optimisation(
    experiment=EXP, weights=[0, 0.5, 0.25, 0.25], preruncycles=cycles, cycles=2, optimisation="GA"
)

POP_Max.optimise()


# In[4]:

EXP.FeMax = POP_Max.exp.FeMax
EXP.FdMax = POP_Max.exp.FdMax


# Below we define two populations of designs.  We will optimise one using the genetic algorithm, and the other using randomly drawn designs.
#
# We optimise for statistical power (weights = [0,1,0,0]).  We run 100 cycles.

# In[5]:

POP_GA = optimisation(
    experiment=EXP,
    weights=[0, 0.5, 0.25, 0.25],
    preruncycles=2,
    cycles=cycles,
    seed=1,
    outdes=5,
    I=10,
    folder="/tmp/",
    optimisation="GA",
)

POP_RN = optimisation(
    experiment=EXP,
    weights=[0, 0.5, 0.25, 0.25],
    preruncycles=2,
    cycles=cycles,
    seed=100,
    outdes=5,
    I=50,
    G=10,
    folder="/tmp/",
    optimisation="simulation",
)


# In[6]:

POP_GA.optimise()


# In[7]:

POP_RN.optimise()


# Below, we show how the efficiency scores improve over cycles for both algorithms, although the Genetic Algorithm clearly improves faster and reaches a higher plateau.

# In[8]:

plt.plot(POP_GA.optima, label="Genetic Algorithm")
plt.plot(POP_RN.optima, label="Simulation")
plt.legend()
plt.savefig(op.join(op.dirname(__file__), "output", "test_scores.pdf"))


# Below, we repeat the random design generator, but we search only 100 designs and one generation.  As such, this is a random design.

# In[9]:

# 1 gen
POP_JO = optimisation(
    experiment=EXP,
    weights=[0, 0.5, 0.25, 0.25],
    preruncycles=1,
    cycles=1,
    seed=1,
    outdes=5,
    G=100,
    folder="/tmp/",
    optimisation="simulation",
)
POP_JO.optimise()


# In[10]:

# collect scores and take average
scores = [x.F for x in POP_JO.designs]

median_idx = np.where(scores == np.median(scores))[0][0]
rnd_median = POP_JO.designs[median_idx]

# get PI
BTI_l = np.percentile(scores, 5)
BTI_u = np.percentile(scores, 95)


# In[11]:

print(
    "Optimisation score - random: %s \nOptimisation score - genetic algorithm: %s \nOptimisation score - simulation (90 percent PI): %s-%s"
    % (POP_RN.optima[::-1][0], POP_GA.optima[::-1][0], BTI_l, BTI_u)
)


# Let's look at the resulting experimental designs.

# In[12]:

des = np.array([POP_GA.bestdesign.Xconv, POP_RN.bestdesign.Xconv, rnd_median.Xconv])
labels = ["Genetic Algorithm", "Simulation", "Median random design"]
plt.figure(figsize=(10, 7))
for ind, label in enumerate(labels):
    plt.subplot(3, 1, ind + 1)
    plt.plot(des[ind, :, :])
    plt.title(label)
    plt.tick_params(axis="x", which="both", bottom="off", labelbottom="off")

plt.savefig("output/designs.pdf")


# In[13]:

des = np.array(
    [POP_GA.bestdesign.Xconv, POP_RN.bestdesign.Xconv] + [x.Xconv for x in POP_JO.designs]
)


# ## Simulate data
#
# We continue with the best designs from the two algorithms and the random design.  Below, we simulate data in one voxel that is significantly related to the task.   We assume beta values of (0.5, 0, -0.5).

# In[ ]:

# create datatables
tp = des.shape[1]
Y = np.zeros([tp, sims, des.shape[0]])

for i in range(sims):
    rnd = np.random.normal(0, 1, tp)
    for lb in range(Y.shape[2]):
        Y[:, i, lb] = np.dot(des[lb, :, :], np.array([0.5, 0, -0.5])) + rnd


# We analyse the data using `R` below.

# In[ ]:

get_ipython().run_cell_magic(
    "R",
    "-i des,Y,sims -o tvals_main,tvals_diff",
    "tvals_main <- array(NA,dim=c(sims,dim(Y)[3]))\ntvals_diff <- array(NA,dim=c(sims,dim(Y)[3]))\nfor (method in 1:dim(Y)[3]){\n    for (sim in 1:sims){\n       dif <- des[method,,1]-des[method,,2]\n        fit <- lm(Y[,sim,method]~des[method,,])\n        tvals_main[sim,method] <- summary(fit)$coef[2,3]\n        fit <- lm(Y[,sim,method]~dif)\n        tvals_diff[sim,method] <- summary(fit)$coef[2,3]\n   }\n}",
)


# This is what the distributions for the two contrasts look like.

# In[ ]:

nms = ["Main effect", "Contrast effect"]
plt.figure(figsize=(18, 4))
dists = [0, 1, median_idx]
for idx, tv in enumerate([tvals_main, tvals_diff]):
    plt.subplot(1, 2, idx + 1)
    for idy, method in enumerate(labels):
        sns.distplot(tv[:, dists[idy]], label=method)
    plt.title(nms[idx])
plt.legend()
plt.savefig("output/distributions.pdf")


# ## Observed power

# In[ ]:

# We're assuming a single threshold on a single test, a representative simplification.
threshold = t.ppf(0.95, des.shape[1] - 2)
nms = ["main effect", "contrast effect"]
out = {label: [] for label in labels}
for idx, tv in enumerate([tvals_main, tvals_diff]):
    for idy, method in enumerate(labels):
        if idy < 2:
            power = np.mean(tv[:, idy] > threshold)
            out[method].append(power)
            print(f"The power for the {nms[idx]} with {method}: {power:f}")
        else:
            powers = [np.mean(tv[:, k] > threshold) for k in range(2, tv.shape[1])]
            out[method].append(powers)
            print(
                "The 90 percent PI for the %s with a randomly drawn design: %f-%f"
                % (nms[idx], np.percentile(powers, 5), np.percentile(powers, 95))
            )
