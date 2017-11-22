import os

os.chdir("/Users/Joke/Documents/Onderzoek/ProjectsOngoing/Neuropower/neurodesign/source/")

from src import geneticalgorithm

EXP = geneticalgorithm.experiment( TR = 3.0, P = [0.33, 0.33, 0.33], C = [[0.5, -0.5, 0.0], [0.5, 0.0, -0.5], [0.0, 0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.5, 0.0, -0.5], [0.0, 0.5, -0.5]], rho = 0.3, n_stimuli = 3, n_trials = 72, duration = 360.0, resolution = 0.1, stim_duration = 3.0, t_pre = 0.0, t_post = 0.0, maxrep = 6, hardprob = False, confoundorder = 3, ITImodel = 'uniform', ITImin = 1.0, ITImean = 2.0, ITImax = 3.0, restnum = 0, restdur = 0.0)

POP = geneticalgorithm.population( experiment = EXP, G = 20, R = [0.4, 0.4, 0.2], q = 0.01, weights = [0.05, 0.8, 0.1, 0.05], I = 4, preruncycles = 10000, cycles = 10000, convergence = 1000, seed = 2574, folder = './')

POP.naturalselection()
