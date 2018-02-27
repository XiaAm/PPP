import numpy as np


class Vertex:
    def __init__(self, _id):
        self._id = id


class Edge:
    def __init__(self, _from, _to, _weight):
        self._from = _from
        self._to = _to
        self._weight = _weight


def generate_graph():
    # see this graph http://blog.csdn.net/qq_35644234/article/details/60870719
    vertex_list = [Vertex('v1'), Vertex('v2'), Vertex('v3'), Vertex('v4'), Vertex('v5'), Vertex('v6')]
    edge_list = [Edge(vertex_list[0], vertex_list[2], 10),
                 Edge(vertex_list[0], vertex_list[4], 30),
                 Edge(vertex_list[0], vertex_list[5], 100),
                 Edge(vertex_list[1], vertex_list[2], 5),
                 Edge(vertex_list[2], vertex_list[3], 50),
                 Edge(vertex_list[3], vertex_list[5], 10),
                 Edge(vertex_list[4], vertex_list[3], 20),
                 Edge(vertex_list[4], vertex_list[5], 60)]
    return vertex_list, edge_list


if True:
    pass
