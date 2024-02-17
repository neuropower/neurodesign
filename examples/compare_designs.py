import neurodesign
from neurodesign import generate

# define experimental setup

exp = neurodesign.experiment(
    TR=2,
    n_trials=20,
    P=[0.3, 0.3, 0.4],
    C=[[1, -1, 0], [0, 1, -1]],
    n_stimuli=3,
    rho=0.3,
    stim_duration=1,
    t_pre=0.5,
    t_post=2,
    ITImodel="exponential",
    ITImin=2,
    ITImax=4,
    ITImean=2.1,
)

# define first design with a fixed ITI

design_1 = neurodesign.design(
    order=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
    ITI=[2] * 20,
    experiment=exp,
)

# expand to design matrix

design_1.designmatrix()
design_1.FCalc(weights=[0, 0.5, 0.25, 0.25])
design_1.FdCalc()
design_1.FcCalc()
design_1.FfCalc()
design_1.FeCalc()

# define second design

design_2 = neurodesign.design(
    order=[0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 1],
    ITI=generate.iti(20, "exponential", min=1, mean=2, max=4, seed=1234)[0],
    experiment=exp,
)

design_2.designmatrix()
design_2.FeCalc()
design_2.FdCalc()
design_2.FcCalc()
design_2.FfCalc()

# crossover to obtain design 3 and 4

design_3, design_4 = design_1.crossover(design_2, seed=2000)
design_3.order
design_4.order
design_3.designmatrix()
design_3.FeCalc()
design_3.FdCalc()
design_3.FcCalc()
design_3.FfCalc()
design_4.designmatrix()
design_4.FeCalc()
design_4.FdCalc()
design_4.FcCalc()
design_4.FfCalc()

# mutate design
DES5 = design_1.mutation(0.3, seed=2000)
DES5.designmatrix()
DES5.FeCalc()
DES5.FdCalc()
DES5.FcCalc()
DES5.FfCalc()

# compare detection power
result = (
    f" RESULTS \n ======= \n"
    f"DESIGN 1: Fd = {design_1.Fd} \n"
    f"DESIGN 2: Fd = {design_2.Fd} \n"
    f"DESIGN 3: Fd = {design_3.Fd} \n"
    f"DESIGN 4: Fd = {design_4.Fd} \n"
    f"DESIGN 5: Fd = {DES5.Fd} \n"
)

print(result)
