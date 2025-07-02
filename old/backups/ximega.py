
### ALGO ###

# Memory for optimized xi and omega functions

xi_memory = {}
omega_memory = {}


# these are the optimized versions of the xi and omega functions they are used to avoid recalculating the same values multiple times
# they use a dictionary to store the results of previous calculations
# the memory is cleared with the clear_memory function, it should be called every time the drones are updated 

def N(i):
    """
    returns the set of neighbors of drone i
    """
    return [n.id for n in Drone.DRONE_REFERENCE[i].get_neighbors()]

def ξ(i, j, n):
    """
    returns 1 if drone i is aware of drone j, 0 otherwise
    """
    if n <= 0: 
        return 1 if i == j else 0
    
    if (i, j, n) in xi_memory:
        return xi_memory[(i, j, n)]
    v = max(ξ(l, j, n-1) for l in N(i) + [i])
    xi_memory[(i, j, n)] = v
    return v

inf = 100000000

def ω(i, j, n):
    """
    returns the smallest amount of edges that connects drone i to drone j
    """
    if n <= 0:
        return 0 if i == j else inf
    
    if (i, j, n) in omega_memory:
        return omega_memory[(i, j, n)]
    
    if ξ(i, j, n-1) == ξ(i, j, n): 
        v = ω(i, j, n-1)
        omega_memory[(i, j, n)] = v
        return v
    else:
        v = min(ω(l, j, n-1) + 1 for l in N(i))
        omega_memory[(i, j, n)] = v
        return v



def Δ(i,j,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    v = ω(i, j, n) - ω(j, l, n)
    return v

def Δ_array(i,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    ret = []
    for j in range(Drone.DRONE_COUNT):
        ret.append(Δ(i,j,l,n))
    return ret




def is_critical_edge(i,l):
    """ returns if the edge i,l is critical """
    n = Drone.DRONE_COUNT
    for j in range(n):
        if Δ(i,j,l,n + 1) == 0:
            return False
        for ii in N(i):
            for ll in N(l):
                if ii == l or ll == i:
                    continue
                if Δ(i,j,ii,n + 1) == 1 and Δ(l,j,ll,n + 1) == 1:
                    return False
    return True
    


########################
#### DISPLAY TABLES ####
########################

