"""
Defines functions for leaky integrate and fire model using Cython
References:
    https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/
    https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
    https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
"""

import numpy as np
cimport numpy as np
cimport cython
from scipy.stats import norm

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def min_max_norm(x):
    '''
    Given an array x, returns x normalized to 0 and 1
    '''
    return (x - np.min(x))/(np.max(x) - np.min(x))


def make_AP(sim, i):
    '''
    Given an instance of Simulation class (sim), 
    and a neuron index (i),
    extracts neuron-specific parameters from sim
    to return a templated action potential

    template from Kakaria and de Bivort (2017)
    '''

    # draw action potential rise (rising part of Gaussian)
    apRiseTime = sim.params['APdurs'][i]/sim.dt/2
    apRise = norm.pdf(np.arange(-1, 0.01, 1/apRiseTime))
    apRise = min_max_norm(apRise)
    apRise = apRise * (sim.params['Vmaxs'][i] - sim.params['Vthrs'][i]) + \
        sim.params['Vthrs'][i]

    # draw action potential fall (fall of sine curve)
    apFallTime = sim.params['APdurs'][i]/sim.dt/2 - 1
    apFall = np.sin( np.arange(np.pi/2, 3*np.pi/2+.001, np.pi/apFallTime))
    apFall = min_max_norm(apFall)
    apFall = apFall * (sim.params['Vmaxs'][i] - sim.params['Vmins'][i]) + \
        sim.params['Vmins'][i]-.0001
        
    # connect rise and fall, and add 2 timepoints of same value
    AP = np.concatenate((apRise, apFall))[1:]
    AP = np.append(AP, AP[-1] + 0.0001)
    AP = np.append(AP, AP[-1] + 0.0001)

    return AP


def make_PSC(sim, i, is_input_PSC=False):
    '''
    Given an instance of Simulation class (sim), 
    and a neuron index (i),
    extracts neuron-specific parameters from sim
    to return a templated post synaptic current

    template from Kakaria and de Bivort (2017)
    '''
    # draw PSC rise (rising part of a sine curve)
    pscRiseT = sim.params['PSCrisedurs'][i]
    pscRise = np.sin(np.linspace(-np.pi/2, np.pi/2, int(pscRiseT/sim.dt)))
    pscRise = min_max_norm(pscRise)
    
    # draw PSC fall (exponential)
    pscFallT = sim.params['PSCfalldurs'][i]
    if is_input_PSC:
        pscFallT = sim.params['inputPSCfalldur']
    pscFall = 2**( -sim.dt/pscFallT * np.arange(7*pscFallT/sim.dt+1))
    pscFall = min_max_norm(pscFall)
    
    # connect rise and fall, and normalize PSC magnitude
    # to counteract changes in duration or time-constants
    PSC = np.concatenate((pscRise, pscFall))    
    PSC = PSC * sim.params['PSCmags'][i] / max(PSC)
    
    return PSC

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef run_LIF_general(np.ndarray[DTYPE_t, ndim=2] connect, 
                    int Nsteps, float dt, 
                    list all_APs, list all_PSCs, 
                    np.ndarray[DTYPE_t, ndim=1] Vthrs, 
                    np.ndarray[DTYPE_t, ndim=1] V0s, 
                    np.ndarray[DTYPE_t, ndim=1] Cmems, 
                    np.ndarray[DTYPE_t, ndim=1] Rmems, 
                    np.ndarray[DTYPE_t, ndim=2] Iin):
    '''
    Run leaky integrate and fire model using Cython.
    
    Returns voltage table (V), current table (I), 
    and action potential counter (APMask), which are
    tables of size (rows: # neurons, columns: # timepoints)

    Code modeled from Kakaria and de Bivort (2017)
    '''
    cdef int numNeurons = len(connect)
    # find a buffer time to add on to the simulation if a PSC
    # begins right before the simulation end time
    cdef list psc_lens = [len(PSC) for PSC in all_PSCs]
    cdef list ap_lens = [len(AP) for AP in all_APs]
    cdef int max_tBuff = max(psc_lens)
    
    # initialize voltages to resting potential
    cdef np.ndarray[DTYPE_t, ndim=2] V = np.tile(V0s, (Nsteps + max_tBuff, 1)).T
    # initialize currents to 0
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.zeros((numNeurons, Nsteps + max_tBuff))
    # initialize action potential mask (i.e. indicator to 
    # not update a neuron if it is in a templated action potential)
    cdef np.ndarray[DTYPE_t, ndim=2] APMask = np.zeros((numNeurons, Nsteps + max_tBuff))
    
    # iterate through time
    cdef int t, i
    cdef np.ndarray APi, PSCi
    cdef int APleni, tBuffi
    cdef float currentTemp, dV
    for t in np.arange(1, Nsteps):
        # iterature through neurons
        for i in range(numNeurons):
            # if this neuron is not masked by an action potential...
            if APMask[i, t] == 0:
                # if it is above spiking threshold...
                if V[i, t-1] > Vthrs[i]:
                    V[i, t-1] = Vthrs[i]
                    # fill in a neuron-specific action potential
                    APi = all_APs[i]; APleni = ap_lens[i]
                    V[i, t:t+APleni] = APi
                    APMask[i, t:t+APleni] = 1
                    # fill in a neuron-specific PSC
                    PSCi = all_PSCs[i]; tBuffi = psc_lens[i]
                    I[i, t:t+tBuffi] = I[i, t:t+tBuffi] + PSCi
                else:
                    # draw input current and current from connectivity
                    currentTemp = Iin[i, t-1] + np.dot(connect[:, i], I[:, t-1])
                    # compute step in voltage, update voltage table
                    dV = 1./Cmems[i] * ( (V0s[i] - V[i, t-1])/Rmems[i] + currentTemp)
                    V[i, t] = V[i, t-1] + dV*dt
    
    # return voltage, current, action potentials
    return V, I, APMask


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef run_LIF_general_sim_wrap(object sim, 
                               np.ndarray[DTYPE_t, ndim=2] Iin):
    '''
    Calls run_LIF_general after passing in Sim object and Iin
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] connect = sim.connect
    cdef int numNeurons = len(connect)

    cdef list all_APs = [make_AP(sim, i) for i in range(numNeurons)]
    cdef list all_PSCs = [make_PSC(sim, i) for i in range(numNeurons)]
    
    cdef int Nsteps = sim.Nsteps
    cdef float dt = sim.dt
    cdef np.ndarray[DTYPE_t, ndim=1] Vthrs = sim.params['Vthrs']
    cdef np.ndarray[DTYPE_t, ndim=1] V0s = sim.params['V0s']
    cdef np.ndarray[DTYPE_t, ndim=1] Cmems = sim.params['Cmems']
    cdef np.ndarray[DTYPE_t, ndim=1] Rmems = sim.params['Rmems']
        
    return run_LIF_general(connect, Nsteps, dt, all_APs, all_PSCs, 
                    Vthrs, V0s, Cmems, Rmems, Iin)


def spikes_from_APMask(APMask):
    '''
    Given a table of action potential indicators (APMask),
    returns a table of same size that marks onsets of action potential
    '''
    Spikes = np.zeros(APMask.shape)
    # for each neuron...
    for i in range(APMask.shape[0]):
        # record indices with action potential mask,
        # and save the first index of each stretch
        row = APMask[i, :]
        where_AP = np.where(row == 1)[0]
        row_spike_times = where_AP[row[where_AP-1] == 0]
        Spikes[i, row_spike_times] = 1
    return Spikes