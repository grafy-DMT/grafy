import numpy as np
from graph import *
from graph import random_directed_graph
from graph import weighted_graph_matrix
from collections import OrderedDict
import copy

stack = 0
visited = 0
components = 0


def dfs_visit(v, graph):
    global stack, visited
    visited[v] = True
    for neighbour in graph.neighbours_lists[v]:
        if not visited[neighbour]:
            dfs_visit(neighbour, graph)
    stack = np.append(stack, v)


def components_r(v, graph):
    global visited, components
    visited[v] = True
    components = np.append(components, v+1)
    for neighbour in graph.neighbours_lists[v]:
        if not visited[neighbour]:
            components_r(neighbour, graph)


def kosaraju(graph):
    global visited, stack, components
    n = graph.vertex_count
    visited = np.zeros(n, bool)
    stack = np.array([], int)
    for vertex in range(n):
        if not visited[vertex]:
            dfs_visit(vertex, graph)
    graph = DirectedGraph(AdjacencyList(transpose_neighbours_lists(graph.neighbours_lists)))
    visited = np.zeros(n, bool)
    cn = 0
    result = []
    while stack.shape[0] != 0:
        v = stack[stack.shape[0] - 1]
        stack = np.delete(stack, stack.shape[0] - 1)
        if not visited[v]:
            cn += 1
            components = np.array([], int)
            components_r(v, graph)
            result.append(components)

    return result


def transpose_neighbours_lists(neighbours_lists):
    new = []
    for i in range(len(neighbours_lists)):
        vertex_neighbours = []
        for vertex in range(len(neighbours_lists)):
            if i in neighbours_lists[vertex]:
                vertex_neighbours.append(vertex)
        new.append(vertex_neighbours)
    return new

# AD3

def random_directed_connected_weighted_graph(vertex_count, edge_probability, weights_range=(1, 10)):
    if edge_probability <= 0:
        raise ValueError("Wrong edge_probability: " + str(edge_probability))
    parts_no = 0
    graph = None
    while parts_no != 1:
        graph = random_directed_weighted_graph(vertex_count, edge_probability, weights_range)
        parts_no = len(kosaraju(graph.directed_graph))

    return graph

d = None
p = None
BIGINT = 922337203

def relax(u, v, graph):
    global d, p
    if d[v] > (d[u] + graph.weights_matrix[u][v]):
        d[v] = d[u] + graph.weights_matrix[u][v]
        p[v] = u

def bellman_fort(graph, s):
    global d, p
    n = graph.vertex_count
    d = np.zeros(n, int)
    p = np.zeros(n, int)
    for vertex in range(n):
        d[vertex] = BIGINT
        p[vertex] = -1
    d[s] = 0
    for _ in range(n):
        for u in range(n):
            for v in graph.neighbours_lists[u]:
                relax(u, v, graph)

    for u in range(n):
        for v in graph.neighbours_lists[u]:
            if d[v] > d[u] + graph.weights_matrix[u][v]:
                return False
    return True

def shortest_paths(graph):
    if type(graph) != DirectedWeightedGraphxD:
        raise ValueError("This function supports only DirectedWeightedGraph")
    result = []
    for i in range(graph.vertex_count):
        if bellman_fort(graph, i):
            result.append(p.copy())
        else:
            return None

    return result

def print_shortest_paths(graph):
    paths = shortest_paths(graph)
    if paths is None:
        print("Found cycle with negative wage")
        return
    print("Shortest paths")
    for u in range(graph.vertex_count):
        for v in range(graph.vertex_count):
            if u == v:
                continue
            stack = []
            tmp = v
            while paths[u][tmp] != -1:
                stack.append(tmp)
                tmp = paths[u][tmp]
            stack.append(u)
            path = ""
            while stack:
                path += str(stack.pop() + 1) + " "
            print(str(u + 1) \
                    + "->"
                    + str(v + 1)
                    + ": "
                    + path)

# AD4

def dijkstra(graph, v=1):
    global d, p
    n = graph.vertex_count
    d = np.zeros(n, int)
    p = np.zeros(n, int)
    Q = np.zeros(n, bool)
    for vertex in range(n):
        d[vertex] = BIGINT
        p[vertex] = -1
        Q[vertex] = False
    d[v - 1] = 0

    S = np.array([], int)

    while S.shape[0] != n:
        j = 0
        while Q[j]:
            j += 1
        u = j
        while j < n:
            if not Q[j] and (d[j] < d[u]):
                u = j
            j += 1
        Q[u] = True
        S = np.append(S, u)
        for neighbour in graph.neighbours_lists[u]:
            if neighbour not in S:
                relax(u, neighbour, graph)

    ptr = 0
    weights = []
    nodes_and_neighbours = OrderedDict()
    for i in range(n):
        weights.append(d[i])
        nodes_and_neighbours[i + 1] = []
        j = i
        while j > -1:
            S[ptr] = j
            ptr += 1
            j = p[j]
        while ptr > 0:
            ptr -= 1
            nodes_and_neighbours[i + 1].append(S[ptr] + 1)
    return weights, nodes_and_neighbours

def add_s(directed_weighted_graph):
    vertex_count = directed_weighted_graph.vertex_count
    tmp_list = copy.deepcopy(directed_weighted_graph.neighbours_lists)
    tmp_list.append([x for x in range(vertex_count)])
    tmp_weights = np.zeros((vertex_count + 1, vertex_count + 1), dtype=int)
    for i in range(vertex_count):
        for j in range(vertex_count):
            tmp_weights[i][j] = directed_weighted_graph.weights_matrix[i][j]

    result = DirectedGraph(AdjacencyList(tmp_list))
    result = DirectedWeightedGraphxD(result, tmp_weights)
    return result



def johnson(directed_weighted_graph):
    global d, p
    vertex_count = directed_weighted_graph.vertex_count
    g_prim = add_s(directed_weighted_graph)
    h = np.zeros((vertex_count + 1), dtype=int)
    if not bellman_fort(g_prim, vertex_count):
        return "Found negative cycle"
    w_daszek = g_prim.weights_matrix.copy()
    for v in range(vertex_count + 1):
        h[v] = d[v]
    for u in range(vertex_count + 1):
        for v in range(vertex_count + 1):
            w_daszek[u][v] = g_prim.weights_matrix[u][v] + h[u] - h[v]
    pogrubione_D = np.zeros((vertex_count, vertex_count), dtype=int)
    save_weights = directed_weighted_graph.weights_matrix
    directed_weighted_graph.weights_matrix = w_daszek
    for u in range(vertex_count):
        dijkstra(directed_weighted_graph, u)
        for v in range(vertex_count):
            pogrubione_D[u][v] = d[v] - h[u] + h[v]
    tmp = pogrubione_D[0].copy()
    for i in range(vertex_count - 1):
        pogrubione_D[i] = pogrubione_D[i+1]
    pogrubione_D[vertex_count - 1] = tmp
    directed_weighted_graph.weights_matrix = save_weights
    return pogrubione_D




def main():
    print("PROJEKCIK 4 GRAFY")

    graph = random_directed_graph(7, 0.4)
    #graph = read_graph_from_file("projekt4/example.txt")
    #graph = DirectedGraph(graph)
    #print(graph)
    component_list = kosaraju(graph)
    print("Silnie spójne składowe:")
    for index, component in enumerate(component_list):
        print(str(index + 1) + ": " + str(component))

    draw_graph(graph)

    print("AD3")

    graph = random_directed_connected_weighted_graph(7, 0.3, (-5, 10))
    print_shortest_paths(graph)
    draw_graph(graph)

    print("AD4")
    print(str(johnson(graph)))
    draw_graph(graph)


if __name__ == "__main__":
    main()
