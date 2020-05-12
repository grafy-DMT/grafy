import numpy as np
from graph import *
from graph import random_directed_graph
from graph import weighted_graph_matrix

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
        bellman_fort(graph, i)
        result.append(p.copy())

    return result

def print_shortest_paths(graph):
    paths = shortest_paths(graph)
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

    graph = random_directed_connected_weighted_graph(7, 0.3, (5, 10))
    print_shortest_paths(graph)
    draw_graph(graph)


if __name__ == "__main__":
    main()
