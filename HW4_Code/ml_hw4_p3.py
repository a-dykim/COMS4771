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

    
def compute_GD(dist_mat, coord, maxiter=1000, learning_rate=0.001, precision=100):
    """
    Compute GD on all cities
    """
    cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
    ncity = len(cities)
    dist_new = compute_distances(coord) ## compute all distance matrix 
    error = np.sqrt(np.sum(dist_mat - dist_new)**2) ## set as the sqrt of discrepancy function
    now = coord.copy()
    old = coord.copy()
    niter = 0
    error_new = 1e5
    while error > precision:
    #and niter < maxiter:
        error_old = error_new
        deltas = {}
        for i in range(ncity):
            fix = np.array(coord[cities[i]])
            #print (cities[i])
            mini = np.zeros((ncity,2))
            for j in range(ncity):
                if i == j:
                    pass
                else:
                    subtract = np.array(coord[cities[j]]) # avoid the subtraction from the same city
                    mini[j] = (dist_new[i][j] - dist_mat[i][j])/dist_new[i][j] * (fix-subtract) 
                    
            deltas[cities[i]] = np.sum(mini, axis=0) # \sum_j (x_i - x_j)
            #print (deltas)
        for k in range(ncity):    
            city = cities[k]
            now[city] = old[city] - learning_rate * deltas[city]
            #print (city)
            #print (now[city])
            #print (deltas[city])
        
        dist_new = compute_distances(now)
        #print (dist_new[0])
        error_new = np.sqrt(np.sum((dist_new - dist_mat)**2))
        old = now
        niter += 1
        print (error_new)
        if error_new > error_old:
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
    

init_x = np.random.random_integers(low=20, high=50, size=9)
init_y = np.random.random_integers(low=-150, high=-60, size=9)
mockcity_dict = {'BOS':[init_x[0], init_y[0]], 'NYC':[init_x[1], init_y[1]], 'DC':[init_x[2], init_y[2]], 'MIA':[init_x[3], init_y[3]],
                 'CHI':[init_x[4], init_y[4]], 'SEA':[init_x[5], init_y[5]], 'SF':[init_x[6], init_y[6]],
                 'LA':[init_x[7], init_y[7]], 'DEN':[init_x[8], init_y[8]]}

actual = {'BOS': array([ 42.3600825, -71.0588801]),
 'CHI': array([41.8781136, -87.6297982]),
 'DC': array([38.9071923,-77.0368707]),
 'DEN': array([39.7392358, -104.990251]),
 'LA': array([34.0522342,  -118.2436849]),
 'MIA': array([25.7616798, -80.1917902]),
 'NYC': array([40.7127837, -74.0059413]),
 'SEA': array([47.6062095,  -122.3320708]),
 'SF': array([37.7749295,-122.4194155])}

result= compute_GD(df_mat,mockcity_dict, learning_rate=0.001, precision=100)
print (result)


