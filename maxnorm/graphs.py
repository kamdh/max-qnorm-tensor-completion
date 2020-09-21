import numpy as np
import networkx as nx
import sparse
import warnings

def obs_mask_iid(shape, density):
    if density > 1:
        warnings.warn("density %f > 1" % density, RuntimeWarning)
        density = 1
    return sparse.random(shape, density)

def obs_mask_expander(graph, t):
    '''
    Construct observations from paths in an expander, as described in the paper.
    '''
    paths = []
    for v in nx.nodes(graph):
        paths.extend(get_paths(graph, v, t-1))
    n = graph.number_of_nodes()
    return sparse.COO(np.array(paths).T, data=1, shape=tuple([n for i in range(t)]))
            
def get_paths(graph, v, depth):
    if depth == 1:
        # return neighbors
        paths = [[v, u] for u in graph[v]]
        return(paths)
    else:
        # DFS
        paths = []
        for n in graph[v]:
            subpaths = get_paths(graph, n, depth-1)
            for u in subpaths:
                paths.append([v, *u])
        return(paths)

# def expander_graph(n, d):
#     '''
#     Construct an expander as a random d-regular graph on n nodes
#     '''
#     return nx.
