import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

__all__ = [
        "AdjacencyList",
        "AdjacencyMatrix",
        "IncidenceMatrix",
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
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    matrix[i][j] -= 1
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
        return AdjacencyList(matrix)
    
def draw_graph(input_graph):
    adjacency_list = convert(input_graph, AdjacencyList)
    # Extract pairs of nodes from adjacency_list
    graph = []
    for node, edges in enumerate(adjacency_list.neighbours_lists, vertex_offset):
        for edge in edges:
            graph.append((node, edge + vertex_offset))

    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # Create NetworkX graph
    G = nx.Graph()
    # Add nodes to graph
    for node in nodes:
        G.add_node(node)

    # Add edges to graph
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # Draw graph on circular layout
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos=pos)
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
        while(randomized_successfully == False):

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
            if (not first_point_available or not second_point_available):
                continue
            first_to_switch = random.choice(list(first_point_available))
            second_to_switch = random.choice(list(second_point_available))

            #removing connection
            neighbours_lists[first_point].remove(first_to_switch)
            neighbours_lists[second_point].remove(second_to_switch)
            neighbours_lists[first_to_switch].remove(first_point)
            neighbours_lists[second_to_switch].remove(second_point)

            #adding new connections
            neighbours_lists[first_point].append(second_to_switch)
            neighbours_lists[second_point].append(first_to_switch)
            neighbours_lists[first_to_switch].append(second_point)
            neighbours_lists[second_to_switch].append(first_point)

            randomized_successfully = True
    return graph

