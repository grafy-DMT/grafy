import numpy as np
from graph import *
from graph import random_directed_graph

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
    print("Silnie spójne składowe:")
    while stack.shape[0] != 0:
        v = stack[stack.shape[0] - 1]
        stack = np.delete(stack, stack.shape[0] - 1)
        if not visited[v]:
            cn += 1
            components = np.array([], int)
            components_r(v, graph)
            print(str(cn) + ": " + str(components))


def transpose_neighbours_lists(neighbours_lists):
    new = []
    for i in range(len(neighbours_lists)):
        vertex_neighbours = []
        for vertex in range(len(neighbours_lists)):
            if i in neighbours_lists[vertex]:
                vertex_neighbours.append(vertex)
        new.append(vertex_neighbours)
    return new


def main():
    print("PROJEKCIK 4 GRAFY")

    graph = random_directed_graph(7, 0.4)
    #graph = read_graph_from_file("projekt4/example.txt")
    #graph = DirectedGraph(graph)
    #print(graph)
    kosaraju(graph)
    draw_graph(graph)


if __name__ == "__main__":
    main()
