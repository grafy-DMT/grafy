import graph as gp
import numpy as np


def hamilton_cycle(v, in_graph, n):
    global stack, visited, test, cycles_found
    stack = np.append(stack, v)

    if stack.shape[0] < n:
        visited[v - 1] = True
        for neighbour in in_graph.neighbours_lists[v - 1]:
            if not visited[neighbour]:
                hamilton_cycle(neighbour + 1, in_graph, n)
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


print("PROJEKCIK 2 GRAFY")

sequence1 = [4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
sequence2 = [4, 4, 3, 1, 2]

print("Sequence1: " + str(gp.is_graph_seq(sequence1)))
print("Sequence2: " + str(gp.is_graph_seq(sequence2)))

gp.draw_graph(gp.graph_from_degree_seq(sequence1))

# HAMILTON CYCLE
graph_in_file = gp.read_graph_from_file('projekt2/ham_ex2.txt')
graph = gp.convert(graph_in_file, gp.AdjacencyList)

# number of hamilton cycles found in a graph
cycles_found = 0
# array of visited vertices
visited = np.zeros(graph.vertex_count, bool)
# storage for hamilton cycle
stack = np.array([], int)
# test if graph is hamiltonian
test = False
print("SZUKANIE CYKLU HAMILTONA")
hamilton_cycle(1, graph, graph.vertex_count)
if cycles_found == 0:
    print("Nie znaleziono cyklu Hamiltona. Podany graf nie jest grafem hamiltonowskim.")
else:
    print("Podany graf jest grafem hamiltonowskim.")
gp.draw_graph(graph)
