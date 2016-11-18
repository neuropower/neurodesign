#!/usr/bin/python2.7
#
# Author: Joke Durnez
#
# Description: Generate m sequences
#
# Loosely translated from http://fmriserver.ucsd.edu/ttliu
#
# Version: 1
#
# Date: 2016-07-13
#
#===========================================================

from __future__ import division
import numpy as np
import math
import pickle
import os
import sys


class Msequence(object):
    '''
    A class for an order of experimental trials.
    '''

    def __init__(self):
        # read in taps file and count
        self.tapsfnc()

    def GenMseq(self,mLen,stimtypeno,seed):
        '''
        Function specific to generate a maximum number of msequences for genetic algorithm.

        :param stimtypeno: Number of different stimulus types
        :type stimtypeno: integer
        :param mLen: The length of the requested msequence (**will be shorter than full msequence**)
        :type MLen: integer
        :param seed: Seed with which msequence is sampled.
        :type seed: integer
        '''

        self.mLen = mLen
        self.stimtypeno = stimtypeno

        # initate baseVal
        baseVal = self.stimtypeno

        # initiate powerVal
        minpow = math.log(mLen+1,baseVal)
        pos = self.taps[baseVal].keys()
        orders = []

        if baseVal == 2:
            # restrict number of possibilities for time constraints,
            # could be lift if analyses are done on HPC
            # still results in 160 msequences...
            pos = pos[:10]

        np.random.seed(seed)
        powerVal = pos[np.random.randint(len(pos))]

        # which sequences are possible
        seqKeys = self.taps[baseVal][powerVal].keys()

        np.random.seed(seed)
        keys = seqKeys[np.random.randint(len(seqKeys))]

        shift = 0
        ms = self.Mseq(baseVal,powerVal,shift,keys)
        if mLen > len(ms):
            rep = np.ceil(mLen/len(ms))
            ms = np.tile(ms,rep)
        if not mLen%len(ms) == 0:
            ms = ms[:mLen]
        ms = [int(x) for x in ms]
        orders.append(ms)

        self.orders = orders
        return self

    def Mseq(self,baseVal,powerVal,shift=None,whichSeq=None,userTaps=None):
        '''
        Function to generate msequences

        :param powerVal: The power of the msequence
        :type powerVal: integer
        :param baseVal: The base value of the msequence (equivalent to number of stimuli)
        :type baseVal: integer
        :param shift: Shift of the msequence
        :type shift: integer
        :param whichSeq: Index of the sequence desired in the taps file.
        :type whichSeq: integer
        :param userTaps: if user wants to specify own polynomial taps
        :type userTaps: list
        '''

        # compute total length
        bitNum = int(baseVal**powerVal-1)

        # initiate register and msequence
        register = np.ones(powerVal)
        ms = np.zeros(bitNum)

        # select possible taps
        tap = self.taps[baseVal][powerVal]

        # if sequence is not given or false : random
        if (not whichSeq) or (whichSeq > len(tap) or whichSeq < 1):
            if whichSeq:
                print("You've asked a non-existing tap ! Generating at random.")
            whichSeq = math.ceil(np.random.uniform(0,1,1)*len(tap))-1

        # generate weights

        if userTaps:
            weights = userTaps
        else:
            weights = np.zeros(powerVal)
            if baseVal == 2:
                tapindex = [x-1 for x in tap[int(whichSeq)]]
                weights[tapindex] = 1
            elif baseVal > 2:
                weights = tap[int(whichSeq)]
            else:
                print("You want at least 2 different stimulus types right? Now you asked for %s"%baseVal)

        # generate msequence
        for i in range(bitNum):
            if baseVal == 4 or baseVal == 8 or baseVal == 9:
                tmp = 0
                for ind in range(len(weights)):
                    tmp = self.qadd(tmp,self.qmult(int(weights[ind]),int(register[ind]),baseVal),baseVal)
                ms[i] = tmp
            else:
                ms[i] = (np.dot(weights,register)+baseVal) % baseVal
            reg_shft = [x for ind,x in enumerate(register) if ind in range(powerVal-1)]
            register=[ms[i]]+reg_shft

        #shift
        if shift == 'random':
            shift = math.ceil(np.random.uniform(0,1,1)*bitNum)-1

        elif shift:
            shift = shift%len(ms)
            ms = np.append(ms[shift:],ms[:shift])

        return ms

    def tapsfnc(self):
        '''
        Function to generate taps leading to msequences.
        '''

        self.taps = {
            # baseval 2
            2:{
                # powerval 2
                2:{
                    1:[1,2]
                },
                # powerval 3
                3:{
                    1:[1,3],
                    2:[2,3]
                },
                # powerval 4
                4:{
                    1:[1,4],
                    2:[3,4]
                },
                5:{
                    1:[2,5],
                    2:[3,5],
                    3:[1,2,3,5],
                    4:[1,2,3,5],
                    5:[1,2,4,5],
                    6:[1,3,4,5]
                },
                6:{
                    1:[1,6],
            		2:[5,6],
            		3:[1,2,5,6],
            		4:[1,4,5,6],
            		5:[1,3,4,6],
            		6:[2,3,5,6],
                },
                7:{
                    1:[1,7],
                    2:[6,7],
                    3:[3,7],
                    4:[4,7],
                    5:[1,2,3,7],
                    6:[4,5,6,7],
                    7:[1,2,5,7],
                    8:[2,5,6,7],
                    9:[2,3,4,7],
                    10:[3,4,5,7],
                    11:[1,3,5,7],
                    12:[2,4,6,7],
                    13:[1,3,6,7],
                    14:[1,4,6,7],
                    15:[2,3,4,5,6,7],
                    16:[1,2,3,4,5,7],
                    17:[1,2,4,5,6,7],
                    18:[1,2,3,5,6,7]
                },
                8:{
                    1:[1,2,7,8],
                    2:[1,6,7,8],
                    3:[1,3,5,8],
                    4:[3,5,7,8],
                    5:[2,3,4,8],
                    6:[4,5,6,8],
                    7:[2,3,5,8],
                    8:[3,5,6,8],
                    9:[2,3,6,8],
                    10:[2,5,6,8],
                    11:[2,3,7,8],
                    12:[1,5,6,8],
                    13:[1,2,3,4,6,8],
                    14:[2,4,5,7,8],
                    15:[1,2,3,6,7,8],
                    16:[1,2,5,6,7,8]
                },
                9:{
                    1:[4,9],
                    2:[5,9],
                    3:[3,4,6,9],
                    4:[3,5,6,9],
                    5:[4,5,8,9],
                    6:[1,4,5,9],
                    7:[1,4,8,9],
                    8:[1,5,8,9],
                    9:[2,3,5,9],
                    10:[4,6,7,9],
                    11:[5,6,8,9],
                    12:[1,3,4,9],
                    13:[2,7,8,9],
                    14:[1,2,7,9],
                    15:[2,4,7,9],
                    16:[2,5,7,9],
                    17:[2,4,8,9],
                    18:[1,5,7,9],
                    19:[1,2,4,5,6,9],
                    20:[3,4,5,7,8,9],
                    21:[1,3,4,6,7,9],
                    22:[2,3,5,6,8,9],
                    23:[3,5,6,7,8,9],
                    24:[1,2,3,4,6,9],
                    25:[1,5,6,7,8,9],
                    26:[1,2,3,4,8,9],
                    27:[1,2,3,7,8,9],
                    28:[1,2,6,7,8,9],
                    29:[1,3,5,6,8,9],
                    30:[1,3,4,6,8,9],
                    31:[1,2,3,5,6,9],
                    32:[3,4,6,7,8,9],
                    33:[2,3,6,7,8,9],
                    34:[1,2,3,6,7,9],
                    35:[1,4,5,6,8,9],
                    36:[1,3,4,5,8,9],
                    37:[1,3,6,7,8,9],
                    38:[1,2,3,6,8,9],
                    39:[2,3,4,5,6,9],
                    40:[3,4,5,6,7,9],
                    41:[2,4,6,7,8,9],
                    42:[1,2,3,5,7,9],
                    43:[2,3,4,5,7,9],
                    44:[2,4,5,6,7,9],
                    45:[1,2,4,5,7,9],
                    46:[2,4,5,6,7,9],
                    47:[1,3,4,5,6,7,8,9],
                    48:[1,2,3,4,5,6,8,9]
                },
                10:{
                    1:[3,10],
                    2:[7,10],
                    3:[2,3,8,10],
                    4:[2,7,8,10],
                    5:[1,3,4,10],
                    6:[6,7,9,10],
                    7:[1,5,8,10],
                    8:[2,5,9,10],
                    9:[4,5,8,10],
                    10:[2,5,6,10],
                    11:[1,4,9,10],
                    12:[1,6,9,10],
                    13:[3,4,8,10],
                    14:[2,6,7,10],
                    15:[2,3,5,10],
                    16:[5,7,8,10],
                    17:[1,2,5,10],
                    18:[5,8,9,10],
                    19:[2,4,9,10],
                    20:[1,6,8,10],
                    21:[3,7,9,10],
                    22:[1,3,7,10],
                    23:[1,2,3,5,6,10],
                    24:[4,5,7,8,9,10],
                    25:[2,3,6,8,9,10],
                    26:[1,2,4,7,8,10],
                    27:[1,5,6,8,9,10],
                    28:[1,2,4,5,9,10],
                    29:[2,5,6,7,8,10],
                    30:[2,3,4,5,8,10],
                    31:[2,4,6,8,9,10],
                    32:[1,2,4,6,8,10],
                    33:[1,2,3,7,8,10],
                    34:[2,3,7,8,9,10],
                    35:[3,4,5,8,9,10],
                    36:[1,2,5,6,7,10],
                    37:[1,4,6,7,9,10],
                    38:[1,3,4,6,9,10],
                    39:[1,2,6,8,9,10],
                    40:[1,2,4,8,9,10],
                    41:[1,4,7,8,9,10],
                    42:[1,2,3,6,9,10],
                    43:[1,2,6,7,8,10],
                    44:[2,3,4,8,9,10],
                    45:[1,2,4,6,7,10],
                    46:[3,4,6,8,9,10],
                    47:[2,4,5,7,9,10],
                    48:[1,3,5,6,8,10],
                    49:[3,4,5,6,9,10],
                    50:[1,4,5,6,7,10],
                    51:[1,3,4,5,6,7,8,10],
                    52:[2,3,4,5,6,7,9,10],
                    53:[3,4,5,6,7,8,9,10],
                    54:[1,2,3,4,5,6,7,10],
                    55:[1,2,3,4,5,6,9,10],
                    56:[1,4,5,6,7,8,9,10],
                    57:[2,3,4,5,6,8,9,10],
                    58:[1,2,4,5,6,7,8,10],
                    59:[1,2,3,4,6,7,9,10],
                    60:[1,3,4,6,7,8,9,10]
                },
                11:{1:[9,11]},
                12:{1:[6,8,11,12]},
                13:{1:[9,10,12,13]},
                14:{1:[4,8,13,14]},
                15:{1:[14,15]},
                16:{1:[4,13,15,16]},
                17:{1:[14,17]},
                18:{1:[11,18]},
                19:{1:[14,17,18,19]},
                20:{1:[17,20]},
                21:{1:[19,21]},
                22:{1:[21,22]},
                23:{1:[18,23]},
                24:{1:[17,22,23,24]},
                25:{1:[22,25]},
                26:{1:[20,24,25,26]},
                27:{1:[22,25,26,27]},
                28:{1:[25,28]},
                29:{1:[27,29]},
                30:{1:[7,28,29,30]}
            },
            3:{
                2:{
                	1:[2,1],
                	2:[1,1]
                },
                3:{
                	1:[0,1,2],
                	2:[1,0,2],
                	3:[1,2,2],
                	4:[2,1,2]
                },
                4:{
                	1:[0,0,2,1],
                	2:[0,0,1,1],
                	3:[2,0,0,1],
                	4:[2,2,1,1],
                	5:[2,1,1,1],
                	6:[1,0,0,1],
                	7:[1,2,2,1],
                	8:[1,1,2,1]
                },
                5:{
                	1:[0,0,0,1,2],
                	2:[0,0,0,1,2],
                	3:[0,0,1,2,2],
                	4:[0,2,1,0,2],
                	5:[0,2,1,1,2],
                	6:[0,1,2,0,2],
                	7:[0,1,1,2,2],
                	8:[2,0,0,1,2],
                	9:[2,0,2,0,2],
                	10:[2,0,2,2,2],
                	11:[2,2,0,2,2],
                	12:[2,2,2,1,2],
                	13:[2,2,1,2,2],
                	14:[2,1,2,2,2],
                	15:[2,1,1,0,2],
                	16:[1,0,0,0,2],
                	17:[1,0,0,2,2],
                	18:[1,0,1,1,2],
                	19:[1,2,2,2,2],
                	20:[1,1,0,1,2],
                	21:[1,1,2,0,2]
                },
                6:{
                	1:[0,0,0,0,2,1],
                	2:[0,0,0,0,1,1],
                	3:[0,0,2,0,2,1],
                	4:[0,0,1,0,1,1],
                	5:[0,2,0,1,2,1],
                	6:[0,2,0,1,1,1],
                	7:[0,2,2,0,1,1],
                	8:[0,2,2,2,1,1],
                	9:[2,1,1,1,0,1],
                	10:[1,0,0,0,0,1],
                	11:[1,0,2,1,0,1],
                	12:[1,0,1,0,0,1],
                	13:[1,0,1,2,1,1],
                	14:[1,0,1,1,1,1],
                	15:[1,2,0,2,2,1],
                	16:[1,2,0,1,0,1],
                	17:[1,2,2,1,2,1],
                	18:[1,2,1,0,1,1],
                	19:[1,2,1,2,1,1],
                	20:[1,2,1,1,2,1],
                	21:[1,1,2,1,0,1],
                	22:[1,1,1,0,1,1],
                	23:[1,1,1,2,0,1],
                	24:[1,1,1,1,1,1]
                },
                7:{
                	1:[0,0,0,0,2,1,2],
                	2:[0,0,0,0,1,0,2],
                	3:[0,0,0,2,0,2,2],
                	4:[0,0,0,2,2,2,2],
                	5:[0,0,0,2,1,0,2],
                	6:[0,0,0,1,1,2,2],
                	7:[0,0,0,1,1,1,2],
                	8:[0,0,2,2,2,0,2],
                	9:[0,0,2,2,1,2,2],
                	10:[0,0,2,1,0,0,2],
                	11:[0,0,2,1,2,2,2],
                	12:[0,0,1,0,2,1,2],
                	13:[0,0,1,0,1,1,2],
                	14:[0,0,1,1,0,1,2],
                	15:[0,0,1,1,2,0,2],
                	16:[0,2,0,0,0,2,2],
                	17:[0,2,0,0,1,0,2],
                	18:[0,2,0,0,1,1,2],
                	19:[0,2,0,2,2,0,2],
                	20:[0,2,0,2,1,2,2],
                	21:[0,2,0,1,1,0,2],
                	22:[0,2,2,0,2,0,2],
                	23:[0,2,2,0,1,2,2],
                	24:[0,2,2,2,2,1,2],
                	25:[0,2,2,2,1,0,2],
                	26:[0,2,2,1,0,1,2],
                	27:[0,2,2,1,2,2,2]
                }
            },
            5:{
                2:{
                	1:[4,3],
                	2:[3,2],
                	3:[2,2],
                	4:[1,3]
                },
                3:{
                	1:[0,2,3],
                	2:[4,1,2],
                	3:[3,0,2],
                	4:[3,4,2],
                	5:[3,3,3],
                	6:[3,3,2],
                	7:[3,1,3],
                	8:[2,0,3],
                	9:[2,4,3],
                	10:[2,3,3],
                	11:[2,3,2],
                	12:[2,1,2],
                	13:[1,0,2],
                	14:[1,4,3],
                	15:[1,1,3]
                },
                4:{
                	1:[0,4,3,3],
                	2:[0,4,3,2],
                	3:[0,4,2,3],
                	4:[0,4,2,2],
                	5:[0,1,4,3],
                	6:[0,1,4,2],
                	7:[0,1,1,3],
                	8:[0,1,1,2],
                	9:[4,0,4,2],
                	10:[4,0,3,2],
                	11:[4,0,2,3],
                	12:[4,0,1,3],
                	13:[4,4,4,2],
                	14:[4,3,0,3],
                	15:[4,3,4,3],
                	16:[4,2,0,2],
                	17:[4,2,1,3],
                	18:[4,1,1,2],
                	19:[3,0,4,2],
                	20:[3,0,3,3],
                	21:[3,0,2,2],
                	22:[3,0,1,3],
                	23:[3,4,3,2],
                	24:[3,3,0,2],
                	25:[3,3,3,3],
                	26:[3,2,0,3],
                	27:[3,2,2,3],
                	28:[3,1,2,2],
                	29:[2,0,4,3],
                	30:[2,0,3,2],
                	31:[2,0,2,3],
                	32:[2,0,1,2],
                	33:[2,4,2,2],
                	34:[2,3,0,2],
                	35:[2,3,2,3],
                	36:[2,2,0,3],
                	37:[2,2,3,3],
                	38:[2,1,3,2],
                	39:[1,0,4,3],
                	40:[1,0,3,3],
                	41:[1,0,2,2],
                	42:[1,0,1,2],
                	43:[1,4,1,2],
                	44:[1,3,0,3],
                	45:[1,3,1,3],
                	46:[1,2,0,2],
                	47:[1,2,4,3],
                	48:[1,1,4,2]
                }
            },
            7:{
                2:{1:[1,4]},
                3:{1:[0,1,5]},
                4:{1:[0,1,1,4]}
            },
            4:{
                2:{1:[1,2]},
                3:{1:[1,1,2]},
                4:{1:[1,2,2,2]}
            },
            8:{
                2:{1:[1,2]},
                3:{1:[1,0,3]},
                4:{1:[1,0,0,5]}
            },
            9:{
                2:{1:[1,3]},
                3:{1:[0,1,3]},
                4:{1:[1,0,0,6]}
            },
            11:{
                2:{
                	1:[10,4],
                	2:[12,4],
                	3:[1,3],
                	4:[3,3],
                	5:[8,3],
                	6:[10,3],
                	7:[12,3],
                	8:[1,4],
                	9:[4,4],
                	10:[7,4],
                	11:[2,5],
                	12:[3,5],
                	13:[8,5],
                	14:[9,5],
                	15:[4,9],
                	16:[5,9],
                	17:[6,9],
                	18:[7,9]
                },
                3:{
                	1:[0,10,7]
                },
                 4:{
                	1:[0,0,10,9]
                }
            },
            13:{
                2:{1:[12,11]},
                3:{1:[0,12,7]}
            }
        }

    @staticmethod
    def qadd(a,b,base):

        if (a >= base or b >= base):
            print('qadd(a,b), a and b must be < %s'%(base))

        if base == 4:
            amat = np.array([
                [0,1,2,3],
            	[1,0,3,2],
            	[2,3,0,1],
            	[3,2,1,0],
            ])
        elif base == 8:
            amat = np.array([
                [0,1,2,3,4,5,6,7],
                [1,0,3,2,5,4,7,6],
                [2,3,0,1,6,7,4,5],
                [3,2,1,0,7,6,5,4],
                [4,5,6,7,0,1,2,3],
                [5,4,7,6,1,0,3,2],
                [6,7,4,5,2,3,0,1],
                [7,6,5,4,3,2,1,0]
            ])
        elif base == 9:
            amat = np.array([
                [0,1,2,3,4,5,6,7,8],
                [1,2,0,4,5,3,7,8,6],
                [2,0,1,5,3,4,8,6,7],
                [3,4,5,6,7,8,0,1,2],
                [4,5,3,7,8,6,1,2,0],
                [5,3,4,8,6,7,2,0,1],
                [6,7,8,0,1,2,3,4,5],
                [7,8,6,1,2,0,4,5,3],
                [8,6,7,2,0,1,5,3,4]
            ])
        else:
            print('qadd base %s not supported yet'%base)

        y = amat[a,b]
        return y

    @staticmethod
    def qmult(a,b,base):

        if (a >= base or b >= base):
            print('qadd(a,b), a and b must be < %s'%(base))

        if base == 4:
            amult = np.array([
                [0,0,0,0],
                [0,1,2,3],
                [0,2,3,1],
                [0,3,1,2]
            ])
        elif base == 8:
            amult = np.array([
                [0,0,0,0,0,0,0,0],
                [0,1,2,3,4,5,6,7],
                [0,2,4,6,5,7,1,3],
                [0,3,6,5,1,2,7,4],
                [0,4,5,1,7,3,2,6],
                [0,5,7,2,3,6,4,1],
                [0,6,1,7,2,4,3,5],
                [0,7,3,4,6,1,5,2]
            ])
        elif base == 9:
            amult = np.array([
                [0,0,0,0,0,0,0,0,0],
                [0,1,2,3,4,5,6,7,8],
                [0,2,1,6,8,7,3,5,4],
                [0,3,6,4,7,1,8,2,5],
                [0,4,8,7,2,3,5,6,1],
                [0,5,7,1,3,8,2,4,6],
                [0,6,3,8,5,2,4,1,7],
                [0,7,5,2,6,4,1,8,3],
                [0,8,4,5,1,6,7,3,2]
        ])
        else:
            print('qmult base %s not supported yet'%base)

        y = amult[a,b]
        return y
