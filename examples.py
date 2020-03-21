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

second_incidence_matrix = gp.convert(neighbour_list, gp.IncidenceMatrix)
print(second_incidence_matrix)



# random graphs

random_graph = gp.random_graph_edge_probability(5, 0.5)
print(random_graph)

# both functions take optional type
random_adjacency_list = gp.random_graph_edge_count(
        10, 10, gp.AdjacencyList)

# random_graph is used to invoke 2 previous functions
third_random_graph = gp.random_graph(
        10, edge_count = 7, graph_type = gp.AdjacencyList)

