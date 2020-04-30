from collections import defaultdict

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

__all__ = [
    "AdjacencyList",
    "AdjacencyMatrix",
    "IncidenceMatrix",
    "WeightedGraph",
    "convert",
    "random_graph",
    "read_graph_from_file",
    "draw_graph"]

# number of first vertex used in displaying and reading data
vertex_offset = 1
edge_offset = vertex_offset


#########################
# GRAPH REPRESENTATIONS
#########################


# this class represents graph as adjacency list
# __init__ takes array of neighbours of 'index' vertex with 
# vertices numbers starting with 0 and array[vertex] being list of 
# neighbours (arrays in 'array' have various length)
class AdjacencyList:

    def __init__(self, array):
        self.vertex_count = len(array)
        self.neighbours_lists = array

    def __str__(self):
        result = "Lista sąsiedztwa\n"
        for vertex in range(self.vertex_count):
            # vertex nr
            result += str(vertex + vertex_offset) + ": "
            # listed neighbours
            result += ", ".join(str(neighbour + vertex_offset)
                                for neighbour in self.neighbours_lists[vertex])
            result += "\n"

        return result


# first dimension is row, second is column
def matrix_to_string(matrix, rows_desc, columns_desc, offset=0):
    max_len_row_desc = max(map(len, rows_desc))
    max_len_col_desc = max(max(map(len, columns_desc)), 2)

    def make_row_desc(string):
        return string.ljust(max_len_row_desc)

    def make_col_desc(string):
        return string.ljust(max_len_col_desc)

    def make_node_desc(number):
        return make_col_desc(str(number + offset))

    def make_separator():
        result = make_row_desc("")
        for col_iter in range(columns_nr):
            result += "+"
            result += "-" * max_len_col_desc

        result += "+\n"
        return result

    rows_nr = len(matrix)
    columns_nr = len(matrix[0])

    result = make_row_desc("") + " " + " ".join(make_col_desc(column)
                                                for column in columns_desc) + "\n"
    result += make_separator()

    for row_iter in range(rows_nr):
        result += make_row_desc(rows_desc[row_iter])
        for col_iter in range(columns_nr):
            result += "|"
            result += make_node_desc(matrix[row_iter][col_iter])

        result += "|\n"
        result += make_separator()

    return result


# matrix member of AdjacencyMatrix is 2 dim matrix with fixed number
# of columns. Each value is either 0 or 1 and tells if two
# vertices are adjacent or not
# matrix[vertex1][vertex2] == matrix[vertex2][vertex1] == 1 =>
# vertex1 and vertex2 are neighbours
class AdjacencyMatrix:

    def __init__(self, array):
        self.vertex_count = len(array)
        self.matrix = array

    def __str__(self):
        global vertex_offset

        columns_and_rows_description = list(map(str,
                                                range(vertex_offset, self.vertex_count + vertex_offset)))

        result = "Macierz sąsiedztwa\n"
        result += matrix_to_string(self.matrix,
                                   columns_and_rows_description,
                                   columns_and_rows_description)

        return result


def adjacency_list_to_adjacency_matrix(adjacency_list):
    vertex_count = adjacency_list.vertex_count
    result_matrix = np.zeros((vertex_count, vertex_count), dtype=int)
    for vertex in range(vertex_count):
        for neighbour in adjacency_list.neighbours_lists[vertex]:
            result_matrix[vertex][neighbour - vertex_offset] = 1

    return AdjacencyMatrix(result_matrix)


def adjacency_matrix_to_adjacency_list(adjacency_matrix):
    vertex_count = adjacency_matrix.vertex_count
    matrix = adjacency_matrix.matrix
    result_list = []
    for vertex_index in range(vertex_count):
        vertex_neighbours = []
        for neighbour_nr in range(vertex_count):
            if bool(matrix[vertex_index][neighbour_nr]):
                vertex_neighbours.append(neighbour_nr)
        result_list.append(vertex_neighbours)

    return AdjacencyList(result_list)


# rows of matrix are vertices, and columns are edges
# if matrix[vertex][edge] == 1 => vertex and edge are incident
class IncidenceMatrix:

    def __init__(self, array):
        self.vertex_count = len(array)
        self.matrix = array
        self.edge_count = len(array[0])

    def __str__(self):
        global vertex_offset

        rows_description = []
        for row_nr in range(self.vertex_count):
            rows_description.append(str(row_nr + vertex_offset))

        edges_description = []
        for edge_nr in range(self.edge_count):
            edges_description.append("L" + str(edge_nr + edge_offset))

        result = "Macierz incydencji\n"
        result += matrix_to_string(self.matrix,
                                   rows_description, edges_description)
        return result


# this class represents weighted graph
# weights_matrix[i][j] = weight of the edge between vertices i and j
# weights are random numbers in the range between 1 and 10
class WeightedGraph:

    def __init__(self, graph):
        if type(graph) != AdjacencyList:
            graph = convert(graph, AdjacencyList)
        self.vertex_count = len(graph.neighbours_lists)
        self.neighbours_lists = graph.neighbours_lists
        self.weights_matrix = weighted_graph_matrix(graph)

    def __str__(self):
        global vertex_offset

        columns_and_rows_description = list(map(str,
                                                range(vertex_offset, self.vertex_count + vertex_offset)))

        result = "Macierz wag\n"
        result += matrix_to_string(self.weights_matrix,
                                   columns_and_rows_description,
                                   columns_and_rows_description)

        return result


def weighted_graph_matrix(graph):
    matrix = np.zeros((graph.vertex_count, graph.vertex_count), dtype=int)

    for vertex in range(graph.vertex_count):
        for neighbour in graph.neighbours_lists[vertex]:
            matrix[vertex][neighbour] = matrix[neighbour][vertex] = random.randint(1, 10)

    return matrix


###############
# CONVERSIONS
###############

def adjacency_matrix_to_incidence_matrix(adjacency_matrix):
    vertex_count = adjacency_matrix.vertex_count
    edge_count = 0
    for vertex_index in range(vertex_count):
        for other_vertex_index in range(vertex_index, vertex_count):
            if bool(adjacency_matrix.matrix[vertex_index][other_vertex_index]):
                edge_count += 1

    result_matrix = np.zeros((vertex_count, edge_count), dtype=int)

    edge_index = 0
    for vertex_index in range(vertex_count):
        for other_vertex_index in range(vertex_index, vertex_count):
            if bool(adjacency_matrix.matrix[vertex_index][other_vertex_index]):
                result_matrix[vertex_index][edge_index] = 1
                result_matrix[other_vertex_index][edge_index] = 1
                edge_index += 1

    return IncidenceMatrix(result_matrix)


def incidence_matrix_to_adjacency_matrix(incidence_matrix):
    vertex_count = incidence_matrix.vertex_count
    edge_count = incidence_matrix.edge_count

    result_matrix = np.zeros((vertex_count, vertex_count), dtype=int)

    for edge_index in range(edge_count):
        vertices = []
        for vertex_index in range(vertex_count):
            if bool(incidence_matrix.matrix[vertex_index][edge_index]):
                vertices.append(vertex_index)

        result_matrix[vertices[0]][vertices[1]] = 1
        result_matrix[vertices[1]][vertices[0]] = 1

    return AdjacencyMatrix(result_matrix)


def adjacency_list_to_incidence_matrix(adjacency_list):
    adjacency_matrix = adjacency_list_to_adjacency_matrix(adjacency_list)
    return adjacency_matrix_to_incidence_matrix(adjacency_matrix)


def incidence_matrix_to_adjacency_list(incidency_matrix):
    adjacency_matrix = incidence_matrix_to_adjacency_matrix(incidency_matrix)
    return adjacency_matrix_to_adjacency_list(adjacency_matrix)


def convert(graph, output_type):
    input_type = type(graph)
    if input_type is output_type:
        return graph
    if input_type is AdjacencyList:
        if output_type is AdjacencyMatrix:
            return adjacency_list_to_adjacency_matrix(graph)
        if output_type is IncidenceMatrix:
            return adjacency_list_to_incidence_matrix(graph)
    if input_type is AdjacencyMatrix:
        if output_type is AdjacencyList:
            return adjacency_matrix_to_adjacency_list(graph)
        if output_type is IncidenceMatrix:
            return adjacency_matrix_to_incidence_matrix(graph)
    if input_type is IncidenceMatrix:
        if output_type is AdjacencyList:
            return incidence_matrix_to_adjacency_list(graph)
        if output_type is AdjacencyMatrix:
            return incidence_matrix_to_adjacency_matrix(graph)
    raise ValueError(
        "Wrong arguments: graph - "
        + repr(graph)
        + ", output_type - "
        + repr(output_type))


#################
# RANDOM GRAPHS
#################

def random_graph_edge_count(
        vertex_count,
        edge_count,
        graph_type=IncidenceMatrix):
    all_pairs = []
    for idx1 in range(vertex_count):
        for idx2 in range(idx1):
            all_pairs.append((idx1, idx2))

    random_pairs = random.sample(all_pairs, edge_count)
    # IncidenceMatrix
    matrix = np.zeros((vertex_count, edge_count), dtype=int)
    for edge_nr in range(edge_count):
        matrix[random_pairs[edge_nr][0]][edge_nr] = 1
        matrix[random_pairs[edge_nr][1]][edge_nr] = 1

    result = IncidenceMatrix(matrix)
    return convert(result, graph_type)


def random_graph_edge_probability(
        vertex_count,
        edge_probability,
        graph_type=AdjacencyMatrix):
    all_pairs = []
    for idx1 in range(vertex_count):
        for idx2 in range(idx1):
            all_pairs.append((idx1, idx2))

    matrix = np.zeros((vertex_count, vertex_count), dtype=int)
    for pair in all_pairs:
        if random.random() < edge_probability:
            matrix[pair[0]][pair[1]] = 1
            matrix[pair[1]][pair[0]] = 1

    result = AdjacencyMatrix(matrix)
    return convert(result, graph_type)


def random_graph(
        vertex_count,
        edge_count=0,
        edge_probability=0,
        graph_type=AdjacencyMatrix):
    if edge_count == 0 and edge_probability == 0:
        raise ValueError("Need to specify edge_count or edge_pobability")

    if edge_probability == 0:
        return random_graph_edge_count(
            vertex_count,
            edge_count,
            graph_type)
    else:
        return random_graph_edge_probability(
            vertex_count,
            edge_probability,
            graph_type)


def read_graph_from_file(filename):
    graph_in_file = AdjacencyMatrix
    with open(filename, 'r') as f:
        try:
            matrix = np.array([line.strip().split() for line in f], int)
        except ValueError:
            f.seek(0)
            matrix = [line.split() for line in f]
            matrix = [list(map(int, i)) for i in matrix]
            graph_in_file = AdjacencyList
    for i in range(len(matrix)):
        for j in matrix[i]:
            if j > 1:
                graph_in_file = AdjacencyList
                break
    if graph_in_file != AdjacencyList:
        if matrix.shape[0] == matrix.shape[1]:
            diagonal = matrix.shape[0]
            for i in range(matrix.shape[0]):
                if matrix[i][i] == 0:
                    diagonal -= 1
            if diagonal == 0 and np.allclose(matrix, matrix.T):
                graph_in_file = AdjacencyMatrix
            else:
                graph_in_file = IncidenceMatrix
        else:
            graph_in_file = IncidenceMatrix

    if graph_in_file == AdjacencyMatrix:
        return AdjacencyMatrix(matrix)
    elif graph_in_file == IncidenceMatrix:
        return IncidenceMatrix(matrix)
    elif graph_in_file == AdjacencyList:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] -= 1
        return AdjacencyList(matrix)


def get_edges_and_nodes_from_adjacency_list(input_graph):
    graph = []
    for node, edges in enumerate(input_graph.neighbours_lists, vertex_offset):
        for edge in edges:
            graph.append((node, edge + vertex_offset))
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])
    return graph, nodes


def draw_graph(input_graph):
    if type(input_graph) != AdjacencyList and type(input_graph) != WeightedGraph:
        input_graph = convert(input_graph, AdjacencyList)
    # Extract pairs of nodes from adjacency_list
    graph = []
    for node, edges in enumerate(input_graph.neighbours_lists, vertex_offset):
        for edge in edges:
            graph.append((node, edge + vertex_offset))
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # Create NetworkX graph
    G = nx.Graph()
    # Add nodes to graph
    for node in nodes:
        G.add_node(node)

    # Add edges to graph
    if type(input_graph) != WeightedGraph:
        for edge in graph:
            G.add_edge(edge[0], edge[1])
    else:
        for edge in graph:
            G.add_edge(edge[0], edge[1], weight=input_graph.weights_matrix[edge[0] - 1][edge[1] - 1])

    # Draw graph on circular layout
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos=pos)

    if type(input_graph) == WeightedGraph:
        # Draw edges weight
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Show graph
    plt.show()


def graph_from_degree_seq(sequence):
    sequence = sequence.copy()
    result_list = [[] for _ in range(len(sequence))]
    for vertex_index in range(len(sequence)):
        vertex_neighbours = result_list[vertex_index]
        neighbour_index = vertex_index + 1
        while sequence[vertex_index] > 0:
            if neighbour_index >= len(sequence):
                raise ValueError("Input is not degree sequence")
            if sequence[neighbour_index] > 0:
                result_list[vertex_index].append(neighbour_index)
                result_list[neighbour_index].append(vertex_index)
                sequence[vertex_index] -= 1
                sequence[neighbour_index] -= 1
            neighbour_index += 1

    return AdjacencyList(result_list)

def is_graph_seq(sequence):
    try:
        graph_from_degree_seq(sequence)
        return True
    except ValueError:
        return False


def randomize_graph(graph, count=1):
    graph = convert(graph, AdjacencyList)
    neighbours_lists = graph.neighbours_lists
    for _ in range(count):
        randomized_successfully = False
        while not randomized_successfully:

            two_random_points = random.sample(
                range(graph.vertex_count), 2)

            first_point = two_random_points[0]
            second_point = two_random_points[1]

            first_point_neighbours = set(neighbours_lists[first_point])
            second_point_neighbours = set(neighbours_lists[second_point])

            # point which we may swap
            first_point_available = first_point_neighbours.copy()
            second_point_available = second_point_neighbours.copy()
            first_point_available.discard(second_point_neighbours)
            first_point_available.discard(second_point)
            second_point_available.discard(first_point_neighbours)
            second_point_available.discard(first_point)

            # no points to switch
            if not first_point_available or not second_point_available:
                continue
            first_to_switch = random.choice(list(first_point_available))
            second_to_switch = random.choice(list(second_point_available))

            # removing connection
            neighbours_lists[first_point].remove(first_to_switch)
            neighbours_lists[second_point].remove(second_to_switch)
            neighbours_lists[first_to_switch].remove(first_point)
            neighbours_lists[second_to_switch].remove(second_point)

            # adding new connections
            neighbours_lists[first_point].append(second_to_switch)
            neighbours_lists[second_point].append(first_to_switch)
            neighbours_lists[first_to_switch].append(second_point)
            neighbours_lists[second_to_switch].append(first_point)

            randomized_successfully = True
    return graph


def components(input_graph):
    # Check that input_graph is adjency list
    component_number = 0
    neighbours = input_graph.neighbours_lists[:]

    # Create another func for max?
    neighbours_size = [len(x) for x in neighbours]
    max_comp = max(neighbours_size)
    max_components = [i for i, j in enumerate(neighbours_size, vertex_offset) if j == max_comp]

    comp = [-1 for _ in range(len(neighbours))]
    for v in range(len(comp)):
        if comp[v] == -1:
            component_number += 1
            comp[v] = component_number
            components_r(component_number, v, neighbours, comp)

    return comp, max_components


def components_r(component_number, v, neighbours, comp):
    for neighbour in neighbours[v]:
        if comp[neighbour] == -1:
            comp[neighbour] = component_number
            components_r(component_number, neighbour, neighbours, comp)


def print_components_and_max_component(input_graph):
    print("---------------------------------------------")
    print("Spojne skladowe i najwkieksza spojna skladowa")
    print("---------------------------------------------")
    comp, max_components = components(input_graph)
    result = defaultdict(list)
    for i, component in enumerate(comp, vertex_offset):
        result[component].append(i)
    for k, v in result.items():
        print(f"{k}) {str(v)[1:-1]}")
    print(str(max_components)[1:-1])


def is_eulerian_graph(input_graph):
    comp, _ = components(input_graph)
    edges, nodes = get_edges_and_nodes_from_adjacency_list(input_graph)
    # Check if there is more than one connected components
    if len(set(comp)) > 1:
        return
    # Check that every node degree is even
    odd_arr = [1 for degree in input_graph.neighbours_lists if len(degree) & 1 != 0]
    if len(odd_arr) > 2:
        return

    eulerian_stack = []
    neighbours = input_graph.neighbours_lists[:]
    neighbour1 = 0
    eulerian_stack.append(neighbour1)

    while not check_if_empty(neighbours):
        try:
            neighbour2 = neighbours[neighbour1].pop(0)
            eulerian_stack.append(neighbour2)
            if neighbour1 in neighbours[neighbour2]:
                neighbours[neighbour2].remove(neighbour1)
            neighbour1 = neighbours[neighbour2].pop(0)
            eulerian_stack.append(neighbour1)
            if neighbour2 in neighbours[neighbour1]:
                neighbours[neighbour1].remove(neighbour2)
        except IndexError:
            return

    return eulerian_stack


def check_if_empty(list_of_lists):
    for elem in list_of_lists:
        if elem:
            return False
    return True


def print_eulerian_path(input_graph):
    path = is_eulerian_graph(input_graph)
    print("------------------------")
    print("Wierzcholki cyklu eulera")
    print("------------------------")
    if path:
        print("[", end="")
        for i in range(len(path) - 1):
            print(path[i] + 1, end=" - ")
        print(f"{path[-1] + 1}]")
    else:
        print("Graf nie jest eulerowski")


def random_graph_degree_count(
        vertex_count,
        degree,
        graph_type=AdjacencyMatrix):
    vertices = int(degree * vertex_count / 2)
    all_pairs = []
    for idx1 in range(vertex_count):
        for idx2 in range(idx1):
            all_pairs.append((idx1, idx2))

    table_of_degrees = []
    for i in range(vertex_count):
        table_of_degrees.append(degree)

    matrix = np.zeros((vertex_count, vertex_count), dtype=int)
    v = vertices
    i = 0
    j = 0
    while i < vertices:
        if (j % 60 == 59):
            i = 0
            table_of_degrees.clear()
            all_pairs.clear()
            for x in range(vertex_count):
                table_of_degrees.append(degree)
            for idx1 in range(vertex_count):
                for idx2 in range(idx1):
                    all_pairs.append((idx1, idx2))
            matrix = np.zeros((vertex_count, vertex_count), dtype=int)

        random.shuffle(all_pairs)
        # print(all_pairs)           POMAGAJĄ ZROZUMIEC DZIALANIE FUNKCJI
        # print(table_of_degrees)    SKUTECZNOŚC 100% ALE TROCHĘ OSZUKANA
        j = j + 1

        if ((table_of_degrees[all_pairs[0][0]] > 0) and (table_of_degrees[all_pairs[0][1]] > 0)):
            matrix[all_pairs[0][0]][all_pairs[0][1]] = 1 + matrix[all_pairs[0][0]][all_pairs[0][1]]
            matrix[all_pairs[0][1]][all_pairs[0][0]] = 1 + matrix[all_pairs[0][1]][all_pairs[0][0]]
            table_of_degrees[all_pairs[0][0]] = table_of_degrees[all_pairs[0][0]] - 1
            table_of_degrees[all_pairs[0][1]] = table_of_degrees[all_pairs[0][1]] - 1
            all_pairs.remove(all_pairs[0])
            i = i + 1

        else:
            random.shuffle(all_pairs)

    result = AdjacencyMatrix(matrix)
    return convert(result, graph_type)


def random_regular_graph(
        vertex_count,
        degree,
        graph_type=AdjacencyMatrix):
    if degree * vertex_count % 2 == 1:
        raise ValueError("degree and vertex count must be even")
    if not 0 <= degree < vertex_count:
        raise ValueError("0 <= degree < vertex count must be satisfied")

    return random_graph_degree_count(
        vertex_count,
        degree,
        graph_type)


# number of hamilton cycles found in a graph
cycles_found = 0


def hamilton_cycle(v, in_graph, n, stack, visited):
    global cycles_found
    stack = np.append(stack, v)

    if stack.shape[0] < n:
        visited[v - 1] = True
        for neighbour in in_graph.neighbours_lists[v - 1]:
            if not visited[neighbour]:
                hamilton_cycle(neighbour + 1, in_graph, n, stack, visited)
        visited[v - 1] = False
    else:
        test = False
        for neighbour in in_graph.neighbours_lists[v - 1]:
            if neighbour == 0:
                test = True
                break
        if test:
            cycles_found += 1
            cycle = "["
            for i in range(stack.shape[0]):
                cycle += str(stack[i]) + " - "
            cycle += str(stack[0]) + "]"
            print(cycle)
    stack = np.delete(stack, stack.shape[0] - 1)
    return cycles_found
