import numpy as np
import pandas as pd
from itertools import permutations 

def get_permutation_pairs(length):
    """
    Compute all permutation pairs
    """
    arr = np.arange(length)
    perm = permutations(arr,2)
    seq = []
    for i in list(perm):
        seq.append(i)
    
    return seq
    
def compute_distances(coordinates):
    """
    Compute L2 distances between all coordinate combinations
    
    Input: coordinates dictionary
    """
    cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
    ncity = len(cities)
    pairs = get_permutation_pairs(ncity)
    distance_matrix = np.zeros((ncity, ncity))
    
    for i in range(len(pairs)):
        loc1 = pairs[i][0]
        loc2 = pairs[i][1]
        city1 = np.array(coordinates[cities[loc1]])
        city2 = np.array(coordinates[cities[loc2]])
        distance_matrix[loc1, loc2] = np.sqrt(np.sum((city1 - city2)**2))
    
    return np.array(distance_matrix)

    
def compute_GD(dist_mat, coord, maxiter=1000, learning_rate=0.001, precision=10000):
    """
    Compute GD on all cities
    """
    cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
    ncity = len(cities)
    dist_new = compute_distances(coord) ## compute all distance matrix 
    error_new = np.sum((dist_mat - dist_new)**2) ## set as the sqrt of discrepancy function
    now = coord.copy()
    old = coord.copy()
    niter = 0
    while error_new > precision: 
#and niter < maxiter:
        error_old = error_new
        deltas = {}
        for i in range(ncity):
            fix = np.array(coord[cities[i]])
            mini = np.zeros((ncity,2))
            for j in range(ncity):
                if i == j:
                    pass
                else:
                    subtract = np.array(coord[cities[j]]) # avoid the subtraction from the same city
                    mini[j] = (dist_new[i][j] - dist_mat[i][j])/dist_new[i][j] * (fix-subtract) 
                    
            deltas[cities[i]] = np.sum(mini, axis=0) # \sum_j (x_i - x_j)
        for k in range(ncity):    
            city = cities[k]
            now[city] = old[city] - learning_rate * deltas[city]
        
        dist_new = compute_distances(now)
        error_new = np.sum((dist_new - dist_mat)**2)
        print (error_new)
        old = now
        niter += 1
        if error_new > error_old:
            return now

    return now 



distances = np.array([['', 'BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN'], ['BOS', 0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949], ['NYC', 206, 0, 233, 1308, 902, 2815, 2934, 2786, 1771], ['DC', 429, 344, 0, 1075, 671, 2684, 2799, 2631, 1616], ['MIA', 1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037], ['CHI', 963, 802, 671, 1329, 0, 2013, 2142, 2054, 996], ['SEA', 2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307], ['SF', 3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235], ['LA', 2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059], ['DEN', 1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])


distance_df = pd.DataFrame(data=distances[1:,1:],
                  index=distances[1:,0],
                  columns=distances[0,1:])

cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
df_mat = np.zeros((9,9))
for i in range(9):
    for j in range(9):
        df_mat[i,j] = int(distance_df[cities[i]][cities[j]])
    
init_dic = {'BOS': [ 0, 0],
 'SF': [-3095, 0],
 'MIA': [-407, 1447],
 'NYC': [-163, 125],
 'DC': [-311, 294],
 'CHI': [-956, 115],
 'SEA': [-2872, -776],              
 'LA': [-2957, 353],
 'DEN': [-1914, -363]}


result= compute_GD(df_mat,init_dic, learning_rate=0.001, precision=10000)
print (result)


