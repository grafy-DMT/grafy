from graph import *
from graph import random_directed_graph


def main():
    print("PROJEKCIK 4 GRAFY")

    graph = random_directed_graph(7, 0.4)
    draw_graph(graph)


if __name__ == "__main__":
    main()
