import numpy as np
from neurodesign import msequence
import scipy.stats as stats
import scipy

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

    if model == "fixed":
        smp = [mean]*ntrials

    elif model == "uniform":
        mean = (min+max)/2.
        maxdur = mean*ntrials-0.5
        success = 0
        ESd = np.sqrt(((max-min)**2/12.)/ntrials)
        while success == 0:
            seed=seed+20
            np.random.seed(seed)
            smp = np.random.uniform(min,max,ntrials)
            if np.sum(smp[1:])<maxdur and (np.mean(smp)-mean)<ESd:
                success = 1

    elif model == "exponential":
        if not lam:
            try:
                lam = compute_lambda(min,max,mean)
            except ValueError as err:
                raise ValueError(err)
        ESd = np.sqrt(lam**2/float(ntrials))
        maxdur = mean*ntrials-0.5
        success = 0
        while success == 0:
            seed = seed+20
            np.random.seed(seed)
            smp = rtexp(ntrials,lam,min,max,seed=seed)
            if np.sum(smp[1:])<maxdur and abs(np.mean(smp)-mean)<(ESd/4.):
                success = 1

    return smp,lam

def compute_lambda(lower,upper,mean):
    a = float(lower)
    b = float(upper)
    m = float(mean)
    opt = scipy.optimize.minimize(difexp,50,args=(a,b,m),bounds=((10**(-9),100),),method="L-BFGS-B")
    return opt.x[0]

def difexp(lam,lower,upper,mean):
    diff = stats.truncexpon((float(upper)-float(lower))/float(lam),loc=float(lower),scale=float(lam)).mean()-float(mean)
    return abs(diff)

def rtexp(ntrials,lam,lower,upper,seed):
    a = float(lower)
    b = float(upper)
    x = lam
    smp = stats.truncexpon((b-a)/x,loc=a,scale=x).rvs(ntrials)
    return smp
