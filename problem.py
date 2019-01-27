import math
import random
import sys
import bisect
from utils import is_in

infinity = float('inf')

class Graph:
    """
    A graph: A -1-> B
               -2-> C
    is represented as g = Graph({'A': {'B': 1, 'C': 2})
    If directed = False, then direction doesn't metter, which means:
            A <-1-> B
              <-2-> C
    Use g.connect('B', 'C', 3) to add more links.
    Use g.nodes() to get a list f nodes. 
    Use g.get('A') to get a dict of links out from A.
    Use g.get('A', 'B') to get the length of the link from A to B.
    """

    def __init__(self, dict = None, directed = True):
        self.dict = dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for state1 in list(self.dict.keys()):
            for (state2, dist) in self.dict[state1].items():
                self.connect1(state2, state1, dist)

    def connect(self, state1, state2, distance = 1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(state1, state2, distance)
        if not self.directed:
            self.connect1(state2, state1, distance)

    def connect1(self, state1, state2, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.dict.setdefault(state1, {})[state2] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)
    
    def get_rev(self, a):
        result = dict()
        for state1 in list(self.dict.keys()):
            for (state2, dist) in self.dict[state1].items():
                if state2 is a:
                    result[state1] = dist
        return result

    def nodes(self):
        """Return a list of nodes in the graph."""
        return list(self.dict.keys())
    
    def __repr__(self):
        return str(self.dict)


def UndirectedGraph(dict = None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(dict = dict, directed = False)


class GraphProblem(object):

    """The problem of searching a graph from initial nodes to another."""

    def __init__(self, initial, goal, graph, hmap = None, locations = None):
        self.initial = initial
        self.goal = goal
        self.graph = graph
        self.hmap = hmap
        self.locs = locations

    def actions(self, state):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(state).keys())
    
    def actions_rev(self, state):
        return list(self.graph.get_rev(state).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action
    
    def cause(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, state1, action, state2):
        return cost_so_far + (self.graph.get(state1, state2) or infinity)
    
    def is_goal(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list) or isinstance(self.goal, set):
            return is_in(state, self.goal)
        else:
            return state == self.goal
    
    def is_start(self, state):
        if isinstance(self.initial, list) or isinstance(self.initial, set):
            return is_in(state, self.initial)
        else:
            return state == self.initial

    def h(self, node):
        """h function is the values stored in hmap"""
        if self.hmap:
            if type(node) is str:
                if node in self.hmap:
                    return self.hmap[node]
                else:
                    return 0
            if node.state in self.hmap:
                return self.hmap[node.state]
            else:
                return 0
            """h function is straight-line distance from a node's state to goal."""
        elif self.locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity

class ArcProblem(GraphProblem):
    """A search problem consists of:
    * a list or set of nodes
    * a list or set of arcs
    * a start node
    * a list or set of goal nodes
    * a dictionary that maps each node into its heuristic value.
    """

    def __init__(self, nodes, arcs, start = None, goals = set(), hmap = {}):
        graph = Graph()
        for node in nodes:
            graph.get(node)
        for arc in arcs:
            graph.connect(arc.from_node, arc.to_node, arc.cost)
        GraphProblem.__init__(self, start, goals, graph, hmap = hmap)
    
    def __repr__(self):
        return str(self.graph)

class Node:
    """ A node:
        1. state
        2. parent: the node that is the successor of
        3. action
        4. path_cost: total path cost 
        5. depth
    """
    def __init__(self, state, parent = None, action = None, path_cost = 0):
        """ Create a node """
        self.state = state
        self.parent = parent
        self.action = action    # the action parent takes
        self.path_cost = path_cost
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
    
    def __repr__(self):
        nodes = self.solution()
        return "<{nodes}>".format(nodes = ", ".join(nodes))
    
    def __lt__(self, node):
        return self.state < node.state
    
    def neighbors(self, problem):
        """ List of neighbor nodes """
        return [self.next_node(problem, action) for action in problem.actions(self.state)]
    
    def neighbors_rev(self, problem):
        return [self.prev_node(problem, action) for action in problem.actions_rev(self.state)]
    
    def prev_node(self, problem, action):
        prev_state = problem.cause(self.state, action)
        return Node(prev_state, self, action, problem.path_cost(self.path_cost, self.state, action, prev_state))
    
    def next_node(self, problem, action):
        """ One node from this node via the given sction"""
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
    
    def connect(self, problem, rev_node):
        prev_state = rev_node.state
        node = rev_node.parent
        connected = self
        while node:
            action = next(filter(lambda action: problem.result(prev_state, action) == node.state, problem.actions(prev_state)), None)
            connected = Node(node.state, connected, action, problem.path_cost(connected.path_cost, connected.state, action, node.state))
            prev_state = node.state
            node = node.parent
        return connected
    
    def reverse(self, problem):
        prev_state = self.state
        node = self.parent
        connected = Node(self.state)
        while node:
            action = next(filter(lambda action: problem.result(prev_state, action) == node.state, problem.actions(prev_state)), None)
            connected = Node(node.state, connected, action, problem.path_cost(connected.path_cost, connected.state, action, node.state))
            prev_state = node.state
            node = node.parent
        return connected
    
    def solution(self):
        """ Start state and List of actions from root to this node"""
        path = self.path
        return [path()[0].state] + [node.action for node in self.path()[1:]]
    
    def path(self):
        """ List of nodes from root to this node"""
        node = self
        rev_path = []
        while node:
            rev_path.append(node)
            node = node.parent
        return list(reversed(rev_path))
    
    def __eq__(self, other):
        """ treat same state as same node """
        return isinstance(other, Node) and self.state == other.state
    
    def __hash__(self):
        return hash(self.state)



    
# an Arc has a from_node, to_node and non-negative cost
class Arc(object):
    def __init__(self, from_node, to_node, cost = 1):
        # cost should not be negative
        assert cost >= 0, ("Cost can not be negative for Arc: <" + str(from_node) + "->" + str(to_node) + ">, cost = " + str(cost))
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost
    
    # string representation of Arc
    def __repr__(self):
        return "<" + str(self.from_node) + "--" + str(self.cost) + "-->" + str(self.to_node) + ">"


    
    


# a Path is a (node/Path + Arc/None)
class Path:
    def __init__(self, path, arc):
        self.path = path
        self.arc = arc
        if arc is None:
            self.cost = 0
        else:
            self.cost = path.cost + arc.cost
    
    # returns the end node of the path
    def end(self):
        if self.arc is None:
            return self.path
        else:
            return self.arc.to_node
    
    # returns all nodes in the path in reverse order
    def nodes(self):
        nodes = []
        current = self
        while current.arc is not None:
            nodes.insert(0, current.arc.to_node)
            current = current.path
        nodes.insert(0, current.path)
        return nodes
    
    def __repr__(self):
        return "<{nodes}>".format(nodes = "->".join(self.nodes()))


cyclic_delivery_problem = ArcProblem(
    {'mail','ts','o103','o109','o111','b1','b2','b3','b4','c1','c2','c3',
     'o125','o123','o119','r123','storage'},
     [  Arc('ts','mail',6), Arc('mail','ts',6),
        Arc('o103','ts',8), Arc('ts','o103',8),
        Arc('o103','b3',4), 
        Arc('o103','o109',12), Arc('o109','o103',12),
        Arc('o109','o119',16), Arc('o119','o109',16),
        Arc('o109','o111',4), Arc('o111','o109',4),
        Arc('b1','c2',3),
        Arc('b1','b2',6), Arc('b2','b1',6),
        Arc('b2','b4',3), Arc('b4','b2',3),
        Arc('b3','b1',4), Arc('b1','b3',4),
        Arc('b3','b4',7), Arc('b4','b3',7),
        Arc('b4','o109',7), 
        Arc('c1','c3',8), Arc('c3','c1',8),
        Arc('c2','c3',6), Arc('c3','c2',6),
        Arc('c2','c1',4), Arc('c1','c2',4),
        Arc('d2','d3',4), Arc('d3','d2',4),
        Arc('d1','d3',8), Arc('d3','d1',8),
        Arc('o125','d2',2), Arc('d2','o125',2),
        Arc('o123','o125',4), Arc('o125','o123',4),
        Arc('o123','r123',4), Arc('r123','o123',4),
        Arc('o119','o123',9), Arc('o123','o119',9),
        Arc('o119','storage',7), Arc('storage','o119',7)],
    start = {'o103'},
    goals = {'r123', 'storage', 'd1'},
    hmap = {
        'mail' : 26,
        'ts' : 23,
        'o103' : 21,
        'o109' : 24,
        'o111' : 27,
        'o119' : 11,
        'o123' : 4,
        'o125' : 6,
        'r123' : 0,
        'b1' : 13,
        'b2' : 15,
        'b3' : 17,
        'b4' : 18,
        'c1' : 6,
        'c2' : 10,
        'c3' : 12,
        'd1' : 8,
        'd2' : 20,
        'd3' : 12,
        'storage' : 12
        }
    )


