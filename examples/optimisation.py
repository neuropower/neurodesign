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
    t_prestim = 0,
    t_poststim = 2,
    restnum=0,
    restdur=0,
    ITImodel = "exponential",
    ITImin = 1,
    ITImean = 2,
    ITImax=4
    )

POP = geneticalgorithm.population(
    experiment=EXP,
    confoundorder=3,
    G = 20,
    R = [0.4,0.4,0.2],
    q = 0.01,
    weights=[0,0.75,0.15,0.1],
    I = 6,
    preruncycles = 2,
    cycles = 2,
    write_score="/Users/Joke/Documents/Onderzoek/Temp/score.txt",
    write_design="/Users/Joke/Documents/Onderzoek/Temp/design.txt",
    statusfile="/Users/Joke/Documents/Onderzoek/Temp/status.txt",
    folder = "/Users/Joke/Documents/Onderzoek/Temp/"
    )

#########################
# run natural selection #
#########################

POP.naturalselection(seed=1)
POP.download()

################
# step by step #
################

POP.max_eff()
POP.add_new_designs(seed=1)
POP.to_next_generation(seed=1)
POP.to_next_generation(seed=1001)

#################
# export report #
#################

report.make_report(POP,"/Users/Joke/hier.pdf")
