import graph as gp

# all these examples are equivalent to graph from classes
neighbour_list = gp.AdjacencyList([
    [1, 4], 
    [0, 2, 3, 4], 
    [1, 3], 
    [1, 2, 4], 
    [0, 1, 3]
    ])
print(neighbour_list)

neighbour_matrix = gp.AdjacencyMatrix([
        [0, 1, 0, 0, 1], 
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0]
        ])
print(neighbour_matrix)

incidence_matrix = gp.IncidenceMatrix([
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0]
        ])
print(incidence_matrix)

