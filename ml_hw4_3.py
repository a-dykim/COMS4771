import numpy as np
import pylab as plt
import pandas as pd

def compute_distance(coord1, coord2):
    """
    Given the two coordinate, compute L2 distance
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    distance = np.sqrt(np.sum((coord1 - coord2)**2))
    return distance

def subtract_wrt_keyword(dic1, dic2):
    """
    Subtract dictionary with same keywords but different values
    """
    keys= list(dic1.keys())
    difference = 0
    #diff_dic = {}
    for i in range(len(keys)):
        val1 = np.array(dic1[keys[i]])
        val2 = np.array(dic2[keys[i]])
        #print (val1-val2)
        difference += np.sqrt(np.sum((val1 - val2)**2))
        #diff_dic[keys[i]] = diff
    return difference # returns to mean difference

def multi_GD_full_rotation(dist_mat, coord, max_totiter=1000, maxiter=1000, learning_rate=0.1, precision=1e-1, change_threshold=0.1):
    """
    Run a GD on all 9 cities full rotation
    
    Input:
    dist_mat : distance matrix (panda Dataframe)
    coordinates : dictionary format(city:[lat,long])
    """
    cities = ['NYC', 'BOS', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
    niter = 0
    change = 1000 # random initial start
    while change > change_threshold:
        print(niter)
    #and niter < max_totiter:
        if niter == 0:
            old = coord.copy()
            new = {}
            new[cities[0]] = coord[cities[0]]

        for i in range(9):
            if i == 8:
                j = 0
            else:
                j = i+1

            city1 = cities[i] ## fixed coordinate
            city2 = cities[j]

            new[city2]= one_GD1(int(dist_mat[city1][city2]), old[city1], old[city2]) # fixed NYC

        changes = subtract_wrt_keyword(new, old)
        old = new.copy()
        niter += 1
    return new

def one_GD1(dist12, coord1, coord2, maxiter=1000, learning_rate=0.1, precision=1e-2):
    """
    Run a gradient descent on two coordinates with respct to coord1
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    initial_dist = compute_distance(coord1, coord2)
    error = np.abs(initial_dist - dist12)# take an arbitary coordinates and compute errors
    niter = 0
    while error > precision and niter < maxiter:
        if niter == 0:
            grad_disc = (initial_dist - dist12) * (coord2- coord1) / initial_dist ## c1 does not change ; initial distance btw two
            c2_old = coord2.copy()
            c2_now = c2_old - learning_rate * grad_disc

        else:
            coord_dist = coord_dist = compute_distance(coord1, c2_old)
            grad_disc = (coord_dist - dist12) * (c2_old- coord1) / coord_dist
            c2_old = c2_now.copy()
            c2_now = c2_old - learning_rate * grad_disc

        error = np.abs(np.sqrt(np.sum((c2_now - coord1)**2)) - dist12)
        niter +=1
        #print (c2_now)
        #discrepancy = (coord_dist - dist12)**2

    return c2_now

distances = np.array([['', 'BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN'], ['BOS', 0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949], ['NYC', 206, 0, 233, 1308, 902, 2815, 2934, 2786, 1771], ['DC', 429, 344, 0, 1075, 671, 2684, 2799, 2631, 1616], ['MIA', 1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037], ['CHI', 963, 802, 671, 1329, 0, 2013, 2142, 2054, 996], ['SEA', 2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307], ['SF', 3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235], ['LA', 2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059], ['DEN', 1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])


distance_df = pd.DataFrame(data=distances[1:,1:],
                  index=distances[1:,0],
                  columns=distances[0,1:])

init = np.random.random_integers(low=0, high=2000, size=16)
mockcity_dict = {'NYC':[0,0], 'BOS':[init[0], init[1]], 'DC':[init[2], init[3]], 'MIA':[init[4], init[5]],
                 'CHI':[init[6], init[7]], 'SEA':[init[8], init[9]], 'SF':[init[10], init[11]],
                 'LA':[init[12], init[13]], 'DEN':[init[14], init[15]]}

result = multi_GD_full_rotation(distance_df,mockcity_dict)
print(result)

