from graph import *


# all these examples are equivalent to graph from classes
neighbour_list = AdjacencyList([
    [1, 4],
    [0, 2, 3, 4],
    [1, 3],
    [1, 2, 4],
    [0, 1, 3]
    ])
print(neighbour_list)

neighbour_matrix = AdjacencyMatrix([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0]
        ])
print(neighbour_matrix)

incidence_matrix = IncidenceMatrix([
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0]
        ])
print(incidence_matrix)

second_incidence_matrix = convert(neighbour_list, IncidenceMatrix)
print(second_incidence_matrix)


# random_graph takes 3 argument: vertex_count, 
# edge_count/edge_probability, graph_type (with default AdjacencyMatrix)
first_random_graph = random_graph(
        10, edge_count = 7, graph_type = AdjacencyList)

second_random_graph = random_graph(
        15, edge_probability = 0.4)

print(second_random_graph)
