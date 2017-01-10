from neurodesign import geneticalgorithm, generate,msequence,report

EXP = geneticalgorithm.experiment(
    TR=2,
    n_trials=100,
    P = [0.33,0.33,0.33],
    C = [[1,0,0],[0,1,0],[0,0,1],[1,-1,0],[0,1,-1]],
    n_stimuli = 3,
    rho = 0.3,
    resolution=1,
    stim_duration=1,
    t_pre = 0,
    t_post = 2,
    restnum=0,
    restdur=0,
    ITImodel = "exponential",
    ITImin = 1,
    ITImean = 2,
    ITImax=4
    )

POP = geneticalgorithm.population(
    experiment=EXP,
    weights=[0,0.5,0.25,0.25],
    preruncycles = 10,
    cycles = 10,
    seed=1,
    folder='/Users/Joke/Documents/Onderzoek/ProjectsOngoing/Neuropower/playground/design'
    )

#########################
# run natural selection #
#########################

POP.naturalselection()
POP.download()
POP.print_cmd()

################
# step by step #
################

POP.add_new_designs(seed=1)
POP.to_next_generation(seed=1)
POP.to_next_generation(seed=1001)

#################
# export report #
#################

report.make_report(POP,"/Users/Joke/hier.pdf")
