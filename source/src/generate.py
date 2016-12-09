import numpy as np
from neurodesign import msequence
import scipy.optimize

def order(nstim,ntrials,probabilities,ordertype,seed=1234):
    '''
    Function will generate an order of stimuli.

    :param nstim: The number of different stimuli (or conditions)
    :type nstim: integer
    :param ntrials: The total number of trials
    :type ntrials: integer
    :param probabilities: The probabilities of each stimulus
    :type probabilities: list
    :param ordertype: Which model to sample from.  Possibilities: "blocked", "random" or "msequence"
    :type ordertype: string
    :param seed: The seed with which the change point will be sampled.
    :type seed: integer or None
    :returns order: A list with the created order of stimuli
    '''
    if ordertype not in ['random','blocked','msequence']:
        raise ValueError(ordertype+' not known.')

    if ordertype == "random":
        np.random.seed(seed)
        mult = np.random.multinomial(1,probabilities,ntrials)
        order = [x.tolist().index(1) for x in mult]

    elif ordertype == "blocked":
        np.random.seed(seed)
        blocksize = float(np.random.choice(np.arange(1,10),1)[0])
        nblocks = int(np.ceil(ntrials/blocksize))
        np.random.seed(seed)
        mult = np.random.multinomial(1,probabilities,nblocks)
        blockorder = [x.tolist().index(1) for x in mult]
        order = np.repeat(blockorder,blocksize)[:ntrials]

    elif ordertype == "msequence":
        order = msequence.Msequence()
        order.GenMseq(mLen=ntrials,stimtypeno=nstim,seed=seed)
        np.random.seed(seed)
        id = np.random.randint(len(order.orders))
        order = order.orders[id]

    return order

def iti(ntrials,model,min=None,mean=None,max=None,lam=None,seed=1234):
    '''
    Function will generate an order of stimuli.

    :param ntrials: The total number of trials
    :type ntrials: integer
    :param model: Which model to sample from.  Possibilities: "fixed","uniform","exponential"
    :type model: string
    :param min: The minimum ITI (required with "uniform" or "exponential")
    :type min: float
    :param mean: The mean ITI (required with "fixed" or "exponential")
    :type mean: float
    :param max: The max ITI (required with "uniform" or "exponential")
    :type max: float
    :param seed: The seed with which the change point will be sampled.
    :type seed: integer or None
    :returns iti: A list with the created ITI's
    '''

    lam = None
    if model == "fixed":
        smp = [mean]*ntrials

    elif model == "uniform":
        mean = (min+max)/2.
        maxdur = mean*ntrials
        success = 0
        while success == 0:
            seed=seed+20
            np.random.seed(seed)
            smp = np.random.uniform(min,max,ntrials)
            if np.sum(smp[1:])<maxdur:
                success = 1

    elif model == "exponential":
        if not lam:
            opt = scipy.optimize.minimize(diftexp,[1.5],args=(min,max,mean,))
            lam = opt.x
        maxdur = mean*ntrials
        success = 0
        while success == 0:
            seed = seed+20
            np.random.seed(seed)
            smp = rtexp(ntrials,lam,min,max,seed=seed)
            if np.sum(smp[1:])<maxdur:
                success = 1

    return smp,lam

def itexp(x,lam,trunc):
    i = -np.log(1-x*(1-np.exp(-trunc*lam)))/lam
    return i

def rtexp(n,lam,min,max,seed):
    trunc = max-min
    np.random.seed(seed)
    r = itexp(np.random.uniform(0,1,n),lam,trunc) + min
    return r

def etexp(lam,min,max,mean):
    trunc = max-min
    exp = 1/lam-trunc*(np.exp(lam*trunc)-1)**(-1)+min
    return exp

def diftexp(lam,min,max,mean):
    exp = etexp(lam,min,max,mean)
    return((exp-mean)**2)
