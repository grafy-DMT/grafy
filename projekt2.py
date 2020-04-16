import graph as gp

print("PROJEKCIK 2 GRAFY")

sequence1 = [4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
sequence2 = [4, 4, 3, 1, 2]

print("Sequence1: " + str(gp.is_graph_seq(sequence1)))
print("Sequence2: " + str(gp.is_graph_seq(sequence2)))

gp.draw_graph(gp.graph_from_degree_seq(sequence1))

