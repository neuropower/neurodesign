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

def iti(ntrials,model,min=None,mean=None,max=None,lam=None,resolution=0.1,seed=1234):
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
    :param resolution: The resolution of the design: for rounding the ITI's
    :type resolution: float
    :param seed: The seed with which the change point will be sampled.
    :type seed: integer or None
    :returns iti: A list with the created ITI's
    '''

    if model == "fixed":
        smp = [0]+[mean]*(ntrials-1)
        smp = resolution*np.round(smp/resolution)

    elif model == "uniform":
        mean = (min+max)/2.
        np.random.seed(seed)
        smp = np.random.uniform(min,max,(ntrials-1))
        smp = _fix_iti(smp,mean,min,max,resolution)
        smp = np.append([0],smp)

    elif model == "exponential":
        if not lam:
            try:
                lam = _compute_lambda(min,max,mean)
            except ValueError as err:
                raise ValueError(err)
        np.random.seed(seed)
        smp = _rtexp((ntrials-1),lam,min,max,seed=seed)
        smp = _fix_iti(smp,mean,min,max,resolution)
        smp = np.append([0],smp)

    # round to resolution

    return smp,lam

def _fix_iti(smp,mean,min,max,resolution):
    # kind of a weird function to fix ITI's to have the nominal mean
    # problem was that you can't just add or subtract the difference: it could be
    # out of bounds of the minimum and the maximum...
    # now it changes values either to min/max or with the average difference
    # compute diff
    smp = resolution*np.round(smp/resolution)
    totaldiff = np.sum(smp) - mean*len(smp)
    while not np.isclose(totaldiff,0,resolution) and np.mean(smp)>mean:
        chid = np.random.choice(len(smp))
        if (smp[chid]-min)<resolution or (max-smp[chid])<resolution:
             continue
        else:
             smp[chid] = smp[chid]-np.sign(totaldiff)*resolution
        totaldiff = np.sum(smp) - mean*len(smp)
    return smp

def _compute_lambda(lower,upper,mean):
    a = float(lower)
    b = float(upper)
    m = float(mean)
    opt = scipy.optimize.minimize(_difexp,50,args=(a,b,m),bounds=((10**(-9),100),),method="L-BFGS-B")
    check = _rtexp(100000,opt.x[0],lower,upper,seed=1000)
    if not np.isclose(np.mean(check),mean,rtol=0.1):
        raise ValueError("Error when figuring out lambda for exponential distribution: can't compute lambda.")
        return o
    else:
        return opt.x[0]

def _difexp(lam,lower,upper,mean):
    diff = stats.truncexpon((float(upper)-float(lower))/float(lam),loc=float(lower),scale=float(lam)).mean()-float(mean)
    return abs(diff)

def _rtexp(ntrials,lam,lower,upper,seed):
    a = float(lower)
    b = float(upper)
    np.random.seed(seed)
    smp = stats.truncexpon((b-a)/lam,loc=a,scale=lam).rvs(ntrials)
    return smp
