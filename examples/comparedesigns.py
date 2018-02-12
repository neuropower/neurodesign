import neurodesign
from neurodesign import generate
import numpy as np

# define experimental setup

EXP = neurodesign.experiment(
    TR=2,
    n_trials=20,
    P = [0.3,0.3,0.4],
    C = [[1,-1,0],[0,1,-1]],
    n_stimuli = 3,
    rho = 0.3,
    stim_duration=1,
    t_pre=0.5,
    t_post=2,
    ITImodel = "exponential",
    ITImin = 2,
    ITImax=4,
    ITImean=2.1
    )

# define first design with a fixed ITI

DES = neurodesign.design(
    order = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1],
    ITI = [2]*20,
    experiment=EXP
)

# expand to design matrix

DES.designmatrix()
DES.FCalc(weights=[0,0.5,0.25,0.25])
DES.FdCalc()
DES.FcCalc()
DES.FfCalc()
DES.FeCalc()

# define second design

DES2 = neurodesign.design(
    order = [0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,2,2,0,1],
    ITI = generate.iti(20,"exponential",min=1,mean=2,max=4,seed=1234)[0],
    experiment=EXP)

DES2.designmatrix(); DES2.FeCalc(); DES2.FdCalc(); DES2.FcCalc(); DES2.FfCalc()

# crossover to obtain design 3 and 4

DES3,DES4 = DES.crossover(DES2,seed=2000)
DES3.order
DES4.order
DES3.designmatrix(); DES3.FeCalc(); DES3.FdCalc(); DES3.FcCalc(); DES3.FfCalc()
DES4.designmatrix(); DES4.FeCalc(); DES4.FdCalc(); DES4.FcCalc(); DES4.FfCalc()

# mutate design
DES5 = DES.mutation(0.3,seed=2000)
DES5.designmatrix(); DES5.FeCalc(); DES5.FdCalc(); DES5.FcCalc(); DES5.FfCalc()

#compare detection power
result = " RESULTS \n" \
         " ======= \n" \
         "DESIGN 1: Fd = {0} \n" \
         "DESIGN 2: Fd = {1} \n" \
         "DESIGN 3: Fd = {2} \n" \
         "DESIGN 4: Fd = {3} \n" \
         "DESIGN 5: Fd = {4} \n".format(DES.Fd,DES2.Fd,DES3.Fd,DES4.Fd,DES5.Fd)

print(result)
