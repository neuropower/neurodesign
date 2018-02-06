from __future__ import division
from neurodesign import msequence, generate, report
from numpy import transpose as t
from scipy.special import gamma
from collections import Counter
from numpy.linalg import inv
from scipy import linalg
import sklearn.cluster
import scipy.linalg
import pandas as pd
import progressbar
import numpy as np
import itertools
import StringIO
import warnings
import zipfile
import shutil
import scipy
import copy
import time
import math
import sys
import os


class design(object):
    '''
    This class represents an experimental design for an fMRI experiment.

    :param order: The stimulus order.
    :type order: list of integers
    :param ITI: The ITI's between all stimuli.
    :type ITI: list of floats
    :param experiment: The experimental setup.
    :type experiment: experiment object
    :param onsets: The onsets of all stimuli.
    :type onsets: list of floats
    '''

    def __init__(self, order, ITI, experiment, onsets=None):

        self.order = order
        self.ITI = ITI
        self.onsets = onsets
        self.Fe = 0
        self.Fd = 0

        self.experiment = experiment

        # assert whether design is valid
        if not len(self.ITI) == experiment.n_trials:
            raise ValueError(
                "length of design (ITI's) does not comply with experiment")
        if not len(self.order) == experiment.n_trials:
            raise ValueError(
                "length of design (orders) does not comply with experiment")

    def check_maxrep(self, maxrep):
        '''
        Function to check whether design does not exceed maximum repeats within design.

        :param maxrep: How many times should a stimulus maximally be repeated.
        :type maxrep: integer
        :returns repcheck: Boolean indicating maximum repeats are respected
        '''
        for stim in range(self.experiment.n_stimuli):
            repcheck = not ''.join(
                str(e) for e in [stim] * maxrep) in ''.join(str(e) for e in self.order)
            if repcheck == False:
                break

        return repcheck

    def check_hardprob(self):
        '''
        Function to check whether frequencies of stimuli are **exactly** the prespecified frequencies.

        :returns probcheck: Boolean indicating probabilities are respected
        '''

        obscnt = Counter(self.order).values()
        obsprob = np.round(obscnt / np.sum(obscnt), decimals=2)
        if not len(self.experiment.P) == len(obsprob):
            probcheck = False
        else:
            close = np.isclose(np.array(self.experiment.P),
                               np.array(obsprob), atol=0.001)
            if not np.sum(close) == len(obsprob):
                probcheck = False
            else:
                probcheck = True

        return probcheck

    def crossover(self, other, seed=1234):
        """
        Function to crossover design with other design and create offspring.

        :param other: The design with which the design will be mixed
        :type other: design object
        :param seed: The seed with which the change point will be sampled.
        :type seed: integer or None
        :returns offspring: List of two offspring designs.
        """

        # check whether designs are compatible
        assert len(self.order) == len(other.order)

        np.random.seed(seed)
        changepoint = np.random.choice(len(self.order), 1)[0]

        offspringorder1 = list(self.order)[
            :changepoint] + list(other.order)[changepoint:]
        offspringorder2 = list(other.order)[
            :changepoint] + list(self.order)[changepoint:]

        offspring1 = design(order=offspringorder1,
                            ITI=self.ITI, experiment=self.experiment)
        offspring2 = design(order=offspringorder2,
                            ITI=other.ITI, experiment=self.experiment)

        return [offspring1, offspring2]

    def mutation(self, q, seed=1234):
        '''
        Function to mutate q% of the stimuli with another stimulus.

        :param q: The percentage of stimuli that should be mutated
        :type q: float
        :param seed: The seed with which the mutation points are sampled.
        :type seed: integer or None
        :returns mutated: Mutated design
        '''

        np.random.seed(seed)
        mut_ind = np.random.choice(len(self.order), int(
            len(self.order) * q), replace=False)
        mutated = copy.copy(self.order)
        for mut in mut_ind:
            np.random.seed(seed)
            mut_stim = np.random.choice(
                self.experiment.n_stimuli, 1, replace=True)[0]
            mutated[mut] = mut_stim

        offspring = design(order=mutated, ITI=self.ITI,
                           experiment=self.experiment)

        return offspring

    def designmatrix(self):
        '''
        Expand from order of stimuli to a fMRI timeseries.
        '''

        # ITIs to onsets
        if self.experiment.restnum > 0:
            orderli = list(self.order)
            ITIli = list(self.ITI)
            for x in np.arange(0, self.experiment.n_trials, self.experiment.restnum)[1:][::-1]:
                orderli.insert(x, "R")
                ITIli.insert(x, self.experiment.restdur)
            ITIli = [y+self.experiment.trial_duration  if not x == "R" else y for x, y in zip(orderli, ITIli)]
            onsets = np.cumsum(ITIli)-self.experiment.trial_duration
            self.onsets = [y for x, y in zip(orderli, onsets) if not x == "R"]
        else:
            ITIli = np.array(self.ITI) + self.experiment.trial_duration
            self.onsets = np.cumsum(ITIli) - self.experiment.trial_duration
        stimonsets = [x + self.experiment.t_pre for x in self.onsets]

        # round onsets to resolution
        self.ITI, x = _round_to_resolution(self.ITI,self.experiment.resolution)
        onsetX, XindStim = _round_to_resolution(stimonsets,self.experiment.resolution)
        stim_duration_tp = int(
            self.experiment.stim_duration / self.experiment.resolution)

        # find indices in resolution scale of stimuli
        assert(np.max(XindStim) <= self.experiment.n_tp)
        assert(np.max(XindStim)+stim_duration_tp <= self.experiment.n_tp)

        # create design matrix in resolution scale (=deltasM in Kao toolbox)
        X_X = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])

        for stimulus in xrange(self.experiment.n_stimuli):
            for dur in xrange(stim_duration_tp):
                X_X[np.array(XindStim) + dur, int(stimulus)
                    ] = [1 if z == stimulus else 0 for z in self.order]

        # deconvolved matrix in resolution units
        deconvM = np.zeros([self.experiment.n_tp, int(
            self.experiment.laghrf * self.experiment.n_stimuli)])
        for stim in xrange(self.experiment.n_stimuli):
            for j in xrange(int(self.experiment.laghrf)):
                deconvM[j:, self.experiment.laghrf * stim +
                        j] = X_X[:(self.experiment.n_tp - j), stim]

        # downsample and whiten deconvM
        idxX = [int(x) for x in np.arange(0, self.experiment.n_tp,
                                         self.experiment.TR / self.experiment.resolution)]
        deconvMdown = deconvM[idxX, :]
        Xwhite = np.dot(
            np.dot(t(deconvMdown), self.experiment.white), deconvMdown)

        # convolve design matrix
        X_Z = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
        for stim in range(self.experiment.n_stimuli):
            X_Z[:, stim] = deconvM[:, (stim * self.experiment.laghrf):(
                (stim + 1) * self.experiment.laghrf)].dot(self.experiment.basishrf)

        # downsample to TR
        idx = [int(x) for x in np.arange(0, self.experiment.n_tp,
                                         self.experiment.TR / self.experiment.resolution)]

        X_Z = X_Z[idx, :]
        X_X = X_X[idx, :]
        Zwhite = t(X_Z) * self.experiment.white * X_Z

        self.X = Xwhite
        self.Z = Zwhite
        self.Xconv = X_Z
        self.Xnonconv = X_X
        self.CX = self.experiment.CX
        self.C = self.experiment.C

        return self

    def FeCalc(self, Aoptimality=True):
        '''
        Compute estimation efficiency.

        :param Aoptimality: Kind of optimality to optimize, A- or D-optimality
        :type Aoptimality: boolean
        '''
        try:
            invM = scipy.linalg.inv(self.X)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.X)
            except np.linalg.linalg.LinAlgError:
                invM = np.nan
        sys.exc_clear()
        invM = np.array(invM)
        st1 = np.dot(self.CX, invM)
        CMC = np.dot(st1, t(self.CX))
        if Aoptimality == True:
            self.Fe = float(self.CX.shape[0] / np.matrix.trace(CMC))
        else:
            self.Fe = float(np.linalg.det(CMC)**(-1 / len(self.C)))
        self.Fe = self.Fe / self.experiment.FeMax
        return self

    def FdCalc(self, Aoptimality=True):
        '''
        Compute detection power.

        :param Aoptimality: Kind of optimality to optimize: A- or D-optimality
        :type Aoptimality: boolean
        '''
        try:
            invM = scipy.linalg.inv(self.Z)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.Z)
            except np.linalg.linalg.LinAlgError:
                invM = np.nan
        sys.exc_clear()
        invM = np.array(invM)
        CMC = np.matrix(self.C) * invM * np.matrix(t(self.C))
        if Aoptimality == True:
            self.Fd = float(len(self.C) / np.matrix.trace(CMC))
        else:
            self.Fd = float(np.linalg.det(CMC)**(-1 / len(self.C)))
        self.Fd = self.Fd / self.experiment.FdMax
        return self

    def FcCalc(self, confoundorder=3):
        '''
        Compute confounding efficiency.

        :param confoundorder: To what order should confounding be protected
        :type confoundorder: integer
        '''
        Q = np.zeros([self.experiment.n_stimuli,
                      self.experiment.n_stimuli, confoundorder])
        for n in xrange(len(self.order)):
            for r in np.arange(1, confoundorder + 1):
                if n > (r - 1):
                    Q[self.order[n], self.order[n - r], r - 1] += 1
        Qexp = np.zeros([self.experiment.n_stimuli,
                         self.experiment.n_stimuli, confoundorder])
        for si in xrange(self.experiment.n_stimuli):
            for sj in xrange(self.experiment.n_stimuli):
                for r in np.arange(1, confoundorder + 1):
                    Qexp[si, sj, r - 1] = self.experiment.P[si] * \
                        self.experiment.P[sj] * (self.experiment.n_trials + 1)
        Qmatch = np.sum(abs(Q - Qexp))
        self.Fc = Qmatch
        self.Fc = 1 - self.Fc / self.experiment.FcMax
        return self

    def FfCalc(self):
        '''
        Compute efficiency of frequencies.
        '''
        trialcount = Counter(self.order)
        Pobs = [trialcount[x] for x in xrange(self.experiment.n_stimuli)]
        self.Ff = np.sum(abs(np.array(
            Pobs) - np.array(self.experiment.n_trials * np.array(self.experiment.P))))
        self.Ff = 1 - self.Ff / self.experiment.FfMax
        return self

    def FCalc(self, weights,Aoptimality=True,confoundorder=3):
        '''
        Compute weighted average of efficiencies.

        :param weights: Weights given to each of the efficiency metrics in this order: Estimation, Detection, Frequencies, Confounders.
        :type weights: list of floats
        '''

        if weights[0]>0:
            self.FeCalc(Aoptimality)
        if weights[1]>0:
            self.FdCalc(Aoptimality)
        self.FfCalc()
        self.FcCalc(confoundorder)
        matr = np.array([self.Fe, self.Fd, self.Ff, self.Fc])
        self.F = np.sum(weights * matr)
        return self


class experiment(object):

    '''
    This class represents an fMRI experiment.

    :param TR: The repetition time.
    :type TR: float
    :param P: The probabilities of each trialtype.
    :type P: ndarray
    :param C: The contrast matrix.  Example: np.array([[1,-1,0],[0,1,-1]])
    :type C: ndarray
    :param rho: AR(1) correlation coefficient
    :type rho: float
    :param n_stimuli: The number of stimuli (or conditions) in the experiment.
    :type n_stimuli: integer
    :param n_trials: The number of trials in the experiment.  Either specify n_trials **or** duration.
    :type n_trials: integer
    :param duration: The total duration (seconds) of the experiment.  Either specify n_trials **or** duration.
    :type duration: float
    :param resolution: the maximum resolution of design matrix
    :type resolution: float
    :param stim_duration: duration (seconds) of stimulus
    :type stim_duration: float
    :param t_pre: duration (seconds) of trial part before stimulus presentation (eg. fixation cross)
    :type t_pre: float
    :param t_post: duration (seconds) of trial part after stimulus presentation
    :type t_post: float
    :param maxrep: maximum number of repetitions
    :type maxrep: integer or None
    :param hardprob: can the probabilities differ from the nominal value?
    :type hardprob: boolean
    :param confoundorder: The order to which confounding is controlled.
    :type confoundorder: integer
    :param restnum: Number of trials between restblocks
    :type restnum: integer
    :param restdur: duration (seconds) of the rest blocks
    :type restdur: float
    :param ITImodel: Which model to sample from.  Possibilities: "fixed","uniform","exponential"
    :type ITImodel: string
    :param ITImin: The minimum ITI (required with "uniform" or "exponential")
    :type ITImin: float
    :param ITImean: The mean ITI (required with "fixed" or "exponential")
    :type ITImean: float
    :param ITImax: The max ITI (required with "uniform" or "exponential")
    :type ITImax: float

    '''

    def __init__(self, TR, P, C, rho, stim_duration, n_stimuli, ITImodel=None, ITImin=None, ITImax=None, ITImean=None, restnum=0, restdur=0, t_pre=0, t_post=0, n_trials=None, duration=None, resolution=0.1, FeMax=1, FdMax=1, FcMax=1, FfMax=1, maxrep=None, hardprob=False, confoundorder=3):
        self.TR = TR
        self.P = P
        self.C = np.array(C)
        self.rho = rho
        self.n_stimuli = n_stimuli
        self.t_pre = t_pre
        self.t_post = t_post
        self.n_trials = n_trials
        self.duration = duration
        self.resolution = resolution
        self.stim_duration = stim_duration

        self.maxrep = maxrep
        self.hardprob = hardprob
        self.confoundorder = confoundorder

        self.ITImodel = ITImodel
        self.ITImin = ITImin
        self.ITImean = ITImean
        self.ITImax = ITImax
        self.ITIlam = None

        self.restnum = restnum
        self.restdur = restdur

        self.FeMax = FeMax
        self.FdMax = FdMax
        self.FcMax = FcMax
        self.FfMax = FfMax

        # make sure resolution is a divisor of TR (up to )
        if not np.isclose(self.TR % self.resolution, 0):
            self.resolution = _find_new_resolution(self.TR,self.resolution)
            warnings.warn("Warning: the resolution is adjusted to be a multiple of the TR.  New resolution: %f"%self.resolution)

        self.countstim()
        self.CreateTsComp()
        self.CreateLmComp()
        self.max_eff()

    def max_eff(self):
        '''
        Function to compute maximum efficiency for Confounding and Frequency efficiency.
        '''
        NulDesign = design(
            order=[np.argmin(self.P)] * self.n_trials,
            ITI=[0]+[self.ITImean] * (self.n_trials-1),
            experiment=self
        )
        NulDesign.designmatrix()
        NulDesign.FcCalc(self.confoundorder)
        self.FcMax = 1 - NulDesign.Fc
        NulDesign.FfCalc()
        self.FfMax = 1 - NulDesign.Ff

        return self

    def countstim(self):
        '''
        Function to compute some arguments depending on other arguments.
        '''
        self.trial_duration = self.stim_duration + self.t_pre + self.t_post

        if self.ITImodel == "uniform":
            self.ITImean = (self.ITImax + self.ITImin) / 2
        if self.duration:
            if not self.restnum == 0:
                # duration of block between rest
                blockdurNR = self.restnum * \
                    (self.ITImean + self.trial_duration)
                blockdurWR = blockdurNR + self.restdur  # duration of block including rest
                # number of blocks
                blocknum = np.floor(self.duration / blockdurWR)
                n_trials = blocknum * self.restnum

                remain = self.duration - (blocknum * blockdurWR)
                if remain >= blockdurNR:
                    n_trials = n_trials + self.restnum
                else:
                    extratrials = np.floor(
                        remain / (self.ITImean + self.trial_duration))
                    n_trials = n_trials + extratrials
                self.n_trials = int(n_trials)
            else:
                self.n_trials = int(
                    self.duration / (self.ITImean + self.trial_duration))
        else:
            ITIdur = self.n_trials * self.ITImean
            TRIALdur = self.n_trials * self.trial_duration
            duration = ITIdur + TRIALdur
            if self.restnum > 0:
                duration = duration + \
                    (np.floor(self.n_trials / self.restnum) * self.restdur)
            self.duration = duration

    def CreateTsComp(self):
        '''
        This function computes the number of scans and timpoints (in seconds and resolution units)
        '''
        self.n_scans = int(np.ceil(self.duration / self.TR))  # number of scans
        # number of timepoints (in resolution)
        self.n_tp = int(np.ceil(self.duration / self.resolution))
        self.r_scans = np.arange(0, self.duration, self.TR)
        self.r_tp = np.arange(0, self.duration, self.resolution)

        return self

    def CreateLmComp(self):
        '''
        This function generates components for the linear model: hrf, whitening matrix, autocorrelation matrix and CX
        '''

        # hrf
        self.canonical()

        # contrasts
        # expand contrasts to resolution
        self.CX = np.array(np.kron(self.C, np.eye(self.laghrf)))
        assert(self.CX.shape[0]==self.C.shape[0]*self.laghrf)
        assert(self.CX.shape[1]==self.n_stimuli*self.laghrf)

        # drift
        self.S = self.drift(np.arange(0, self.n_scans))  # [tp x 1]
        assert(self.S.shape==(3,self.n_scans))
        self.S = np.matrix(self.S)

        # square of the whitening matrix
        base = [1 + self.rho**2, -1 * self.rho] + [0] * (self.n_scans - 2)
        self.V2 = scipy.linalg.toeplitz(base)
        # set first and last to 1
        self.V2[0, 0] = 1
        self.V2[self.n_scans - 1, self.n_scans - 1] = 1
        self.V2 = np.matrix(self.V2)

        self.white = self.V2 - self.V2 * \
            t(self.S) * np.linalg.pinv(self.S *
                                       self.V2 * t(self.S)) * self.S * self.V2

        return self

    def canonical(self):
        '''
        This function generates the canonical hrf

        :param resolution: resolution to sample the canonical hrf
        :type resolution: float
        '''
        # translated from spm_hrf
        p = [6, 16, 1, 1, 6, 0, 32]
        dt = self.resolution
        s = np.array(xrange(int(np.ceil(p[6] / dt))))
        # HRF sampled at resolution
        hrf = self.spm_Gpdf(s, p[0] / p[2], dt / p[2]) - \
            self.spm_Gpdf(s, p[1] / p[3], dt / p[3]) / p[4]
        self.basishrf = hrf / np.sum(hrf)
        s# duration of the HRF
        self.durhrf = p[6]
        # length of the HRF parameters in resolution scale
        self.laghrf = int(np.ceil(self.durhrf / self.resolution))
        assert(self.laghrf == len(s))

        return self

    @staticmethod
    def drift(s, deg=3):
        '''
        Function to compute a drift component
        '''
        S = np.ones([deg, len(s)])
        s = np.array(s)
        tmpt = np.array(2. * s / float(len(s) - 1) - 1)
        S[1] = tmpt
        for k in np.arange(2, deg):
            S[k] = ((2. * k - 1.) / k) * tmpt * S[k - 1] - \
                ((k - 1) / float(k)) * S[k - 2]
        return S

    @staticmethod
    def spm_Gpdf(s, h, l):
        '''
        Function to generate gamma pdf
        '''
        s = np.array(s)
        res = (h - 1) * np.log(s) + h * np.log(l) - l * s - np.log(gamma(h))
        return np.exp(res)


class population(object):
    '''
    This class represents the population of experimental designs for fMRI.

    :param experiment: The experimental setup of the fMRI experiment.
    :type experiment: experiment
    :param G: The size of the generation
    :type G: integer
    :param R: with which rate are the orders generated from ['blocked','random','mseq']
    :type R: list
    :param q: percentage of mutations
    :type q: float
    :param weights: weights attached to Fe, Fd, Ff, Fc
    :type weights: list
    :param I: number of immigrants
    :type I: integer
    :param preruncycles: number of prerun cycles (to find maximum Fe and Fd)
    :type preruncycles: integer
    :param cycles: number of cycles
    :type cycles: integer
    :param seed: seed
    :type seed: integer
    :param Aoptimality: optimises A-optimality if true, else D-optimality
    :type Aoptimality: boolean
    :param convergence: after how many stable iterations is there convergence
    :type convergence: integer
    :param folder: folder to save output
    :type folder: string
    :param outdes: number of designs to be saved
    :type outdes: integer
    :param optimisation: The type of optimisation - 'GA' or 'random'
    :type optimisation: string
    '''

    def __init__(self, experiment, weights, preruncycles, cycles, seed=None, I=4, G=20, R=[0.4, 0.4, 0.2], q=0.01, Aoptimality=True, folder=None, outdes=3, convergence=1000,optimisation='GA'):

        self.exp = experiment
        self.G = G
        self.R = R
        self.q = q
        self.weights = weights
        self.I = I
        self.preruncycles = preruncycles
        self.cycles = cycles
        self.convergence = convergence
        self.Aoptimality = Aoptimality
        self.outdes = outdes
        self.folder = folder
        self.optimisation = optimisation
        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(10000)

        self.designs = []
        self.optima = []
        self.bestdesign = None
        self.cov = None

    def change_seed(self):
        '''
        Function to change the seed.
        '''
        if self.seed < 4 * 10**9:
            self.seed = self.seed + 1000
        else:
            self.seed = 1

        return self

    def check_develop(self, design, weights=None):
        '''
        Function to check and develop a design to the population.  Function will check design against strict options and develop the design if valid.

        :param design: Design to be added to population.
        :type design: design object
        :param weights: weights for efficiency calculation.
        :type weights: list of floats, summing to 1
        '''
        # weights

        if weights == None:
            weights = self.weights

        # check maxrep, hardprob, every stimulus at least once
        if not self.exp.maxrep == None:
            if not design.check_maxrep(self.exp.maxrep):
                return False
        if self.exp.hardprob:
            if not design.check_hardprob():
                return False
        if len(np.unique(design.order)) < self.exp.n_stimuli:
            return False

        # develop

        out = design.designmatrix()
        if out == False:
            return False
        design.FCalc(weights,confoundorder=self.exp.confoundorder,Aoptimality=self.Aoptimality)
        if np.isnan(design.F):
            return False

        return design

    def add_new_designs(self, weights=None, R=None):
        '''
        This function generates the population.

        :param experiment: The experimental setup of the fMRI experiment.
        :type experiment: experiment
        :param weights: weights for efficiency calculation.
        :type weights: list of floats, summing to 1
        :param seed: The seed for ramdom processes.
        :type seed: integer or None
        '''
        # weights
        if weights == None:
            weights = self.weights

        if not R:
            R = np.round(np.array(self.R) * self.G).tolist()

        if self.exp.n_stimuli in [6, 10] and R[2] > 0:
            print("warning: for this number of conditions/stimuli, there are no msequences possible.  Replaced by random designs.")
            R[1] = R[1] + R[2]
            R[2] = 0

        NDes = 0
        self.change_seed()

        k = 0
        while NDes < np.sum(R):
            self.change_seed()
            ind = np.sum(NDes >= np.cumsum(R))
            ordertype = ['blocked', 'random', 'msequence'][ind]

            order = generate.order(self.exp.n_stimuli, self.exp.n_trials,
                                   self.exp.P, ordertype=ordertype, seed=self.seed)
            ITI,ITIlam = generate.iti(ntrials=self.exp.n_trials, model=self.exp.ITImodel, min=self.exp.ITImin, max=self.exp.ITImax, mean=self.exp.ITImean, lam=self.exp.ITIlam, seed=self.seed,resolution=self.exp.resolution)

            if ITIlam:
                self.exp.ITIlam = ITIlam

            des = design(order=order, ITI=ITI, experiment=self.exp)

            fulldes = self.check_develop(des, weights)
            if fulldes == False:
                continue
            else:
                self.designs.append(fulldes)
                NDes = NDes + 1

        return self

    def _clean_designs(self,weights,seed):
        n = 0
        rm = 0
        while n == 0:
            orders = [x.order for x in self.designs]
            cors = np.corrcoef(orders)
            isone = np.isclose(cors, 1.)
            if len(isone) == 1:
                n = 1
            else:
                np.fill_diagonal(isone, 0)
                if np.sum(isone) == 0:
                    n = 1
                else:
                    ind = np.where(isone)
                    remove = ind[1][ind[0] == ind[0][0]]
                    self.designs = [des for ind, des in enumerate(
                        self.designs) if not ind in remove]
                    rm = rm + len(remove)

        self.add_new_designs(R=[0, rm, 0], weights=weights)

        return self

    def _mutation(self,weights,seed):
        # Mutation:
        # if: Best design: stay untouched
        # elif Correlation between all is > 0.8: mutate with 20% mutations
        # else: mutate with 5% mutations
        # for all: if conditions are not fulfilled: not mutated

        signals = [x.Xconv for x in self.designs]
        efficiencies = [x.F for x in self.designs]

        cors = self.pearsonr(signals, self.exp.n_stimuli)
        mncor = np.mean(cors)

        for idx in range(len(self.designs)):
            design = self.designs[idx]

            if design.F == np.max(efficiencies):
                offspring = design

            elif mncor > 0.6:
                offspring = design.mutation(0.2, seed=seed)
                offspring = self.check_develop(offspring, weights)

            else:
                offspring = design.mutation(self.q, seed=seed)
                offspring = self.check_develop(offspring, weights)

            if offspring == False:
                continue
            else:
                self.designs[idx] = offspring

        return self

    def _crossover(self,weights,seed):
        # select designs with F>median(F):
        efficiencies = [x.F for x in self.designs]
        #crossind = [ind for ind,val in enumerate(efficiencies) if val >= np.median(efficiencies)]
        crossind = range(len(self.designs))

        nparents = int(len(crossind))
        npairs = int(nparents / 2.)

        np.random.seed(seed)
        CouplingRnd = np.random.choice(
            nparents, size=(npairs * 2), replace=False)
        CouplingRnd = [crossind[x] for x in CouplingRnd]
        CouplingRnd = [[CouplingRnd[i], CouplingRnd[i + 1]]
                       for i in np.arange(0, npairs * 2, 2)]

        count = 0

        for couple in CouplingRnd:
            baby1, baby2 = self.designs[couple[0]].crossover(
                self.designs[couple[1]], seed=seed)
            for baby in [baby1, baby2]:
                baby = self.check_develop(baby, weights)
                if baby == False:
                    continue
                else:
                    self.designs.append(baby)
                    count = count + 1

        return self

    def _immigration(self,weights,noim):
        R = np.ceil(np.array(self.R) * noim).tolist()
        self.add_new_designs(R=R, weights=weights)

        return self

    def to_next_generation(self, weights=None, seed=1234,optimisation=None):
        '''
        This function goes from one generation to the next.

        :param weights: weights for efficiency calculation.
        :type weights: list of floats, summing to 1
        :param seed: The seed for random processes.
        :type seed: integer or None
        :param optimisation: The type of optimisation - 'GA' or 'random'
        :type optimisation: string
        '''

        if optimisation == None:
            optimisation = self.optimisation

        # weights
        if weights == None:
            weights = self.weights

        self._clean_designs(weights,seed)

        # remove duplicates and replace by random designs
        if optimisation == 'GA':
            self._mutation(weights,seed)
            self._crossover(weights,seed)
            self._immigration(weights,noim=self.I)

        elif optimisation == 'random':
            self._immigration(weights,noim=self.I)

        else:
            print("Unknown optimisation type")

        # inspect efficiencies
        efficiencies = [x.F for x in self.designs]
        maximum = np.max(efficiencies)
        self.optima.append(maximum)
        bestind = [ind for ind, val in enumerate(
            efficiencies) if val == maximum][0]
        self.bestdesign = self.designs[bestind]

        # append best designs to lists

        # check convergence
        gen = len(self.optima)
        if gen > 1000:
            if self.optima[-1] > self.optima[gen - 1000]:
                self.finished = True

        # select best G
        cutoff = np.sort(efficiencies)[::-1][self.G]
        self.designs = [des for ind, des in enumerate(
            self.designs) if des.F >= cutoff]

        return self

    def clear(self):
        '''
        Function to clear results between optimalisations (maximum Fe, Fd or opt)
        '''
        self.designs = []
        self.optima = []
        self.finished = False
        self.change_seed()

        if self.bestdesign:
            bestdes = design(order=self.bestdesign.order,
                             ITI=self.bestdesign.ITI, experiment=self.exp)
            bestdes = self.check_develop(bestdes)
            if not bestdes == False:
                self.designs.append(bestdes)
            self.bestdesign = None

        return self

    def naturalselection(self,optimisation='GA'):
        '''
        Function to run natural selection for design optimization
        '''

        if (self.exp.FcMax == 1 and self.exp.FfMax == 1):
            self.exp.max_eff()

        if self.exp.FeMax == 1 and self.weights[0] > 0:
            # add new designs
            self.clear()
            self.add_new_designs(weights=[1, 0, 0, 0])
            # loop
            bar = progressbar.ProgressBar()
            for generation in bar(range(self.preruncycles)):
                self.to_next_generation(seed=self.seed, weights=[1, 0, 0, 0])
                if self.finished:
                    continue
            self.exp.FeMax = np.max(self.bestdesign.F)

        if self.exp.FdMax == 1 and self.weights[1] > 0:
            self.clear()
            self.add_new_designs(weights=[0, 1, 0, 0])
            # loop
            bar = progressbar.ProgressBar()
            for generation in bar(range(self.preruncycles)):
                self.to_next_generation(seed=self.seed, weights=[0, 1, 0, 0])
                if self.finished:
                    continue
            self.exp.FdMax = np.max(self.bestdesign.F)

        # clear all attributes
        self.clear()
        self.add_new_designs()
        # loop
        bar = progressbar.ProgressBar()
        for generation in bar(range(self.cycles)):
            self.to_next_generation(seed=self.seed)
            if self.finished:
                    continue

        return self

    def evaluate(self):
        # select designs: best from k-means clusters
        shape = self.bestdesign.Xconv.shape
        xdim = np.zeros(np.product(shape))
        des = np.zeros([np.product(shape), len(self.designs)])
        efficiencies = np.array([x.F for x in self.designs])
        for d in range(len(self.designs)):
            hrf = []
            for stim in range(shape[1]):
                hrf = hrf + self.designs[d].Xconv[:, stim].tolist()
            des[:, d] = hrf
        clus = sklearn.cluster.k_means(des.T, self.outdes,random_state=self.seed)[1]
        out = []
        des = []
        cl = []
        first = 0
        for c in range(self.outdes):
            ids = np.where(clus==c)[0]
            id_ordered = ids[np.flipud(np.argsort(efficiencies[ids]))]
            out.append(first)
            for d in id_ordered:
                cl.append(c)
                des.append(self.designs[d])
                first = first+1
        self.designs = des
        self.out = out
        self.clus = cl

        signals = [x.Xconv for x in self.designs]
        co = self.pearsonr(signals,3)
        self.cov = co

        return self

    def download(self):
        if not self.folder:
            raise ValueError('No folder defined to download output.')
        else:
            if self.cov==None:
                self.evaluate()

            # empty folder
            if os.path.exists(self.folder):
                files = os.listdir(self.folder)
                for f in files:
                    if 'design_' in f:
                        shutil.rmtree(os.path.join(self.folder, f))
            else:
                os.mkdir(self.folder)

            reportfile = "report.pdf"
            report.make_report(self, os.path.join(self.folder, reportfile))

            files = []

            for des in range(self.outdes):

                os.mkdir(os.path.join(self.folder, "design_" + str(des)))

                design = self.designs[self.out[des]]

                for stim in range(self.exp.n_stimuli):

                    onsetsfile = os.path.join(
                        "design_" + str(des), "stimulus_" + str(stim) + ".txt")

                    onsubsets = [str(x) for x in np.array(design.onsets)[
                        np.array(design.order) == stim]]
                    f = open(os.path.join(self.folder, onsetsfile), 'w+')
                    for line in onsubsets:
                        f.write(line)
                        f.write("\n")
                    f.close()

                    files.append(onsetsfile)

                itifile = os.path.join("design_" + str(des), "ITIs.txt")

                f = open(os.path.join(self.folder, itifile), 'w+')
                for line in design.ITI:
                    f.write(str(line))
                    f.write("\n")
                f.close()

                files.append(itifile)
            files.append(reportfile)

            # zip up
            zip_subdir = "OptimalDesign"
            self.zip_filename = "%s.zip" % zip_subdir
            self.file = StringIO.StringIO()
            zf = zipfile.ZipFile(self.file, "w")

            for fpath in files:
                zf.write(os.path.join(self.folder, fpath),
                         os.path.join(zip_subdir, fpath))
            zf.close()

            return self

    @staticmethod
    def pearsonr(signals, nstim):
        cor = []
        varcov = np.zeros([len(signals), len(signals)])
        for sig1 in range(len(signals)):
            for sig2 in range(sig1, len(signals)):
                cors = np.diag(np.corrcoef(
                    t(signals[sig1]), t(signals[sig2]))[nstim:, :nstim])
                varcov[sig1, sig2] = np.mean(cors)
                varcov[sig2, sig1] = np.mean(cors)
        return varcov

def _change_resolution(inputmatrix,start=1,goal=0.1):#for example
    newmat = inputmatrix/goal

def _find_new_resolution(TR,res):
    n = TR*1000.
    # find divisors of TR*1000
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            large_divisors.append(i)
            if i*i != n:
                large_divisors.append(int(n / i))
    sorted = np.sort(large_divisors)
    # closest to res
    resdivisor = TR/float(res)
    difs = np.abs(resdivisor-sorted)
    minind = np.where(difs==np.min(difs))[0]
    divisor = sorted[minind][0]
    newres = TR/divisor
    return newres

def _round_to_resolution(inmat,res):
    out = res*np.floor(np.array(inmat)/res)
    ind = out/res
    ind = [int(x) for x in ind]
    return out, ind
