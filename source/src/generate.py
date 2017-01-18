import numpy as np
from neurodesign import msequence
import scipy.stats as stats

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
    check = False
    while check == False:

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
                try:
                    lam = compute_lambda(min,max,mean)
                except:
                    '''what to do?'''
            maxdur = mean*ntrials
            success = 0
            while success == 0:
                seed = seed+20
                np.random.seed(seed)
                smp = rtexp(ntrials,lam,min,max,seed=seed)
                if np.sum(smp[1:])<maxdur:
                    success = 1

    return smp,lam

def compute_lambda(lower,upper,mean):
    a = float(lower)
    b = float(upper)
    m = float(mean)
    res = 1
    rng = np.arange(10**(-10),1000,res)
    done = False
    while done == False:
        diff=[]
        for x in rng:
            diff.append(stats.truncexpon((b-a)/x,loc=a,scale=x).mean()-m)
        diff = np.array(diff)
        if np.all(diff>0):
            raise ValueError('Impossible to sample from this truncated exponential distribution.  The mean is too far from the maximum.  Either decrease the maximum or increase the mean.')
            done = True
        elif np.all(diff<0):
            raise ValueError('Impossible to sample from this truncated exponential distribution.  The mean is too close to the maximum.  The maximum should be higher than twice the mean.  Either increase the maximum or decrease the mean.')
            done = True
        else:
            if res<10**(-10):
                out = rng[idx]
                done = True
            idx = np.min(np.where(diff>0))
            res = res/10.
            rng = np.arange(rng[idx-1],rng[idx],res)
    return out

def rtexp(ntrials,lam,lower,upper,seed):
    a = float(lower)
    b = float(upper)
    x = lam
    smp = stats.truncexpon((b-a)/x,loc=a,scale=x).rvs(ntrials)
    return smp
