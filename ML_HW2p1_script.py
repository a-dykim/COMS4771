### Gradient Descent Algorithm for P1 ###

import numpy as np
import pylab as plt

def GD_regression(x,y,a0=0,b0=0, maxiter=1000, learning_rate=0.01, precision=0.00001):
    """
    Fit a straight line ax+b using Gradient Descent
    """
    N = len(y)
    costs = np.zeros(maxiter)
    error = 1
    niter = 0
    while error > precision and niter<maxiter:
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y.copy()
        
        y_previous = y_now
        y_now = (a_now * x) + b_now
        error = np.abs(sum(y_now - y_previous))
        costs[niter] = sum([c**2 for c in (y-y_now)]) / N
        grad_a = -(2/N) * sum(x*(y-y_now))
        grad_b = -(2/N) * sum(y-y_now)      
        a_now = a_now - (learning_rate * grad_a)
        b_now = b_now - (learning_rate * grad_b)
        niter += 1
        y_previous = y_now
        
    return a_now, b_now, costs, niter


def SGD_regression(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.01, precision=1e-4):
    """
    Fit a straight line ax+b with SGD (smaller subsets of data for each iteration)
    """
    N = len(y)
    costs = np.zeros(maxiter)
    niter = 0
    error = 1
    
    while error > precision and niter<maxiter:
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]

        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
            
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        error = np.abs(sum(y_now - y_previous))
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)      
        a_now = a_now - (learning_rate * grad_a)
        b_now = b_now - (learning_rate * grad_b)
        niter += 1
        
    return a_now, b_now, costs, niter


def SGD_momentum_regression(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.001, vrate=0.25, precision=1e-4):
    """
    Fit a straight line ax+b with SGD Moment(smaller subsets of data for each iteration)
    vrate: velocity weight
    """
    N = len(y)
    costs = np.zeros(maxiter)
    vel_a = np.zeros(maxiter)
    vel_b = np.zeros(maxiter)
    niter = 0
    error = 1
    
    while error > precision and niter < maxiter:
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]
        
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
            
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        error = np.abs(sum(y_now - y_previous))
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)
        a_past = a_now ## just to perserve
        b_past = b_now
        
        a_now = a_now - (learning_rate * grad_a + vrate * vel_a[niter])
        b_now = b_now - (learning_rate * grad_b + vrate * vel_b[niter])
        
        vel_a[niter] = a_now - a_past
        vel_b[niter] = b_now - b_past
        niter += 1
    
    velocities = np.vstack((vel_a, vel_b))
        
    return a_now, b_now, costs, velocities, niter


def AdaGrad(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.01, precision=1e-4):
    """
    Fit a straight line ax+b with SGD Moment(smaller subsets of data for each iteration)
    vrate: velocity weight
    """
    N = len(y)
    costs = np.zeros(maxiter)
    niter = 0
    error = 1
    
    while error > precision and niter < maxiter:
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]
        
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
            
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        error = np.abs(sum(y_now - y_previous))
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)
        
        eps = 0.1 * np.std(x) ## small noise term to fiddle around
        a_now = a_now - (learning_rate * grad_a) / (np.sqrt(grad_a**2)+eps)
        b_now = b_now - (learning_rate * grad_b) / (np.sqrt(grad_b**2)+eps)
        niter += 1
        
    return a_now, b_now, costs, niter


def RMSprop(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.01, gamma=0.85, precision=1e-4):
    """
    Fit a straight line ax+b with SGD Moment(smaller subsets of data for each iteration)
    gamma: weights of how much previous gradient is considered
    """
    N = len(y)
    costs = np.zeros(maxiter)
    a_decay = np.zeros(maxiter+1)
    b_decay = np.zeros(maxiter+1)
    niter = 0
    error = 1
    
    while error > precision and niter < maxiter:            
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
        
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        error = np.abs(sum(y_now - y_previous))
        
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)

        eps = 0.1 * np.std(y_now) ## small noise term to fiddle around
        
        if niter == 0:
            a_decay[niter], b_decay[niter] = grad_a**2, grad_b**2
          
        a_decay[niter+1] =  gamma * a_decay[niter] + (1-gamma) * (grad_a**2)
        b_decay[niter+1] = gamma * b_decay[niter] + (1-gamma) * (grad_b**2) 
        
        a_now = a_now - (learning_rate * grad_a) / (np.sqrt(a_decay[niter+1])+eps)
        b_now = b_now - (learning_rate * grad_b) / (np.sqrt(b_decay[niter+1])+eps)

        niter += 1
        
    return a_now, b_now, costs, niter



def AdaDelta(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.01, gamma=0.85, precision=1e-4):
    """
    Fit a straight line ax+b with SGD Moment(smaller subsets of data for each iteration)
    gamma: weights of how much previous gradient is considered
    """
    N = len(y)
    costs = np.zeros(maxiter)
    a_decay = np.zeros(maxiter+1)
    b_decay = np.zeros(maxiter+1)
    niter = 0
    error = 1
    
    while error > precision and niter < maxiter:   
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
        
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)
        error = np.abs(sum(y_now - y_previous))

        eps = 0.1 * np.std(x) ## small noise term to fiddle around
        
        if niter == 0:
            a_decay[niter], b_decay[niter] = grad_a**2, grad_b**2
        
        a_now = a_now - (learning_rate * grad_a) / (np.sqrt(a_decay[niter]**2)+eps)
        b_now = b_now - (learning_rate * grad_b) / (np.sqrt(b_decay[niter]**2)+eps)

        a_decay[niter+1] =  gamma * a_decay[niter] + (1-gamma) * (grad_a**2)
        b_decay[niter+1] = gamma * b_decay[niter] + (1-gamma) * (grad_b**2) 
        niter +=1
        
    return a_now, b_now, costs, niter



def ADAM(x,y,a0=0,b0=0, subset_size=20, maxiter=10000, learning_rate=0.01, gamma1=0.45, gamma2=0.65, precision=1e-4):
    """
    Fit a straight line ax+b with SGD Moment(smaller subsets of data for each iteration)
    gamma: weights of how much previous gradient is considered
    """
    N = len(y)
    costs = np.zeros(maxiter)
    a_decay1 = np.zeros(maxiter+1)
    b_decay1 = np.zeros(maxiter+1)
    a_decay2 = np.zeros(maxiter+1)
    b_decay2 = np.zeros(maxiter+1)
    niter = 0
    error = 1
    
    while error > precision and niter < maxiter:  
        
        subset_index = np.random.random_integers(low=0, high=N-1, size=subset_size)
        x_subset = x[[subset_index]]
        y_subset = y[[subset_index]]
        if niter == 0:
            a_now, b_now = a0, b0
            y_now = y_subset.copy()
            
        y_previous = y_now
        y_now = (a_now * x_subset) + b_now
        costs[niter] = sum([c**2 for c in (y_subset-y_now)]) / N
        error = np.abs(sum(y_now - y_previous))
        grad_a = -(2/N) * sum(x_subset*(y_subset-y_now))
        grad_b = -(2/N) * sum(y_subset-y_now)

        eps = 0.1 * np.std(x) ## small noise term to fiddle around
        
        if niter == 0:
            a_decay1[niter], b_decay1[niter] = grad_a**2, grad_b**2
            a_decay2[niter], b_decay2[niter] = grad_a, grad_b
        
        a_now = a_now - (learning_rate * a_decay1[niter]) / (np.sqrt(a_decay2[niter]**2)+eps)
        b_now = b_now - (learning_rate * b_decay1[niter]) / (np.sqrt(b_decay2[niter]**2)+eps)

        a_decay1[niter+1] =  gamma1 * a_decay1[niter] + (1-gamma1) * (grad_a)
        b_decay1[niter+1] = gamma1 * b_decay1[niter] + (1-gamma1) * (grad_b) 
        a_decay2[niter+1] =  gamma2 * a_decay2[niter] + (1-gamma2) * (grad_a**2)
        b_decay2[niter+1] = gamma2 * b_decay2[niter] + (1-gamma2) * (grad_b**2)
        
        niter += 1
        
    return a_now, b_now, costs, niter


