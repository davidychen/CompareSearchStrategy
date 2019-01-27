from utils import *
from problem import *
import pprint
import copy

class Searcher(object):
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    This does depth-first search unless overridden
    """
    
    def __init__(self, problem, method, depth_limit = -1, pruning = 'cycle', bidir = False,  max_expanded=10000):
        """max_expanded is a bound on the number of paths expanded (to prevent infinite computation)
        """
        self.method_map = {
            'Depth-first': {
                'frontier': lambda: Stack()
            },
            'Breadth-first': {
                'frontier': lambda: FIFOQueue()
            },
            'Lowest-cost-first': {
                'f':        lambda node: node.path_cost,
                'frontier': lambda: PriorityQueue(min, self.f), 
                'best':     True
            },
            'Iterative deepening': {
                'frontier': lambda: Stack()
            },
            'Depth-limited': {
                'frontier': lambda: Stack()
            },
            'Heuristic depth-first': {
                'local':    True,
                'local_f':  lambda node: self.problem.h(node),
                'local_q':  lambda: PriorityQueue(min, self.local_f),
                'frontier': lambda: Stack()
            },
            'Greedy best-first': {
                'f':        lambda node: self.problem.h(node),
                'frontier': lambda: PriorityQueue(min, self.f), 
                'best':     True
            },
            'A*': {
                'f':        lambda node: node.path_cost + self.problem.h(node),
                'frontier': lambda: PriorityQueue(min, self.f), 
                'best':     True
            }, 
            'Bidirectional': {
                'bidir': True
            }
        }
        self.problem = problem
        self.method = method
        self.pruning = pruning
        self.bidir = bidir
        self.depth_limit = depth_limit
        self.max_expanded = max_expanded
        self.initialize()
    
    def initialize(self):
        # set f
        self.f = self.method_map.get(self.method, {'f': lambda x: x}).get('f')
        # set frontier
        self.frontier = self.method_map.get(self.method, {'frontier': lambda: []}).get('frontier', lambda: [])()
        # set best
        self.best = self.method_map.get(self.method, {'best': False}).get('best')
        # set pruning
        self.graph = (self.pruning == 'graph') or ('graph' in self.pruning)
        self.cycle = (self.pruning == 'cycle') or ('cycle' in self.pruning)
        # local
        self.local = self.method_map.get(self.method, {'local': False}).get('local')
        if self.local:
            self.local_f = self.method_map.get(self.method, {'local_f': lambda x: x}).get('local_f')
            self.local_q = self.method_map.get(self.method, {'local_q': lambda: []}).get('local_q')()
        if self.bidir:
            self.frontierB = copy.deepcopy(self.frontier)
        self.num_expanded = 0
        self.iterative_reset()

    def iterative_reset(self):
        self.frontier.clear()
        if self.bidir:
            self.frontierB.clear()
        # set beginning node(s)
        if isinstance(self.problem.initial, list) or isinstance(self.problem.initial, set):
            #forEach(lambda state: self.frontier.append(Node(state)), self.problem.initial)
            for state in self.problem.initial:
                self.frontier.append(Node(state))
        else:
            self.frontier.append(Node(self.problem.initial))
        
        if self.bidir:
            self.frontierB.clear()
            if isinstance(self.problem.goal, list) or isinstance(self.problem.goal, set):
                #forEach(lambda state: self.frontier.append(Node(state)), self.problem.initial)
                for state in self.problem.goal:
                    self.frontierB.append(Node(state))
            else:
                self.frontierB.append(Node(self.problem.goal))
        self.explored = set()
        self.depth_drop = False
    
    def search_all(self):
        self.iterative_reset()
        if self.method != 'Iterative deepening':
            result = []
            while self.num_expanded < self.max_expanded:
                solution = self.search()
                if solution == None:
                    break
                if not self.bidir:
                    result += [solution]
                else:
                    if solution.solution() not in map(lambda node: node.solution(), result):
                        result += [solution]
            return result
        else:
            self.depth_limit = 0
            result = []
            temp_expanded = 0
            last_new_expanded = 0
            new_expanded = 0
            temp_depth = 0
            while self.num_expanded < self.max_expanded:
                solution = self.search()
                new_expanded += self.num_expanded - temp_expanded
                temp_expanded = self.num_expanded
                if temp_depth != self.depth_limit:
                    if temp_depth + 1 == self.depth_limit and new_expanded == last_new_expanded and not self.depth_drop:
                        break
                    temp_depth = self.depth_limit
                    last_new_expanded = new_expanded
                    new_expanded = 0
                    self.depth_drop = False
                if solution.solution() not in map(lambda node: node.solution(), result):
                    result += [solution]
            return result
    
    def search(self):
        if self.method != 'Iterative deepening':
            if self.bidir:
                return self.bidirectional_search()
            else:
                return self.search1()
        else:
            self.depth_limit = max(0, self.depth_limit)
            while self.depth_limit < sys.maxsize:
                if self.num_expanded >= self.max_expanded:
                    break
                result = self.search1()
                if result != None:
                    return result
                else:
                    self.iterative_reset()
                    self.depth_limit += 1
    
    def search1(self):
        while self.frontier and self.num_expanded < self.max_expanded:
            node = self.frontier.pop()
            printdb(0, "Expanding:", node, "| cost =", node.path_cost)
            self.num_expanded += 1
            if not self.cycle or node.state not in node.solution()[:-1]:
                if self.problem.is_goal(node.state):
                    printdb(0, "Explored:", self.num_expanded, ",", len(self.frontier), "remained,", node, "found")
                    return node
                if self.graph:
                    self.explored.add(node.state)
                for child in node.neighbors(self.problem):
                    if not self.graph or (child.state not in self.explored and child not in self.frontier):
                        if self.depth_limit < 0 or child.depth <= self.depth_limit:
                            if self.local:
                                self.local_q.append(child)
                            else:
                                self.frontier.append(child)
                        else:
                            self.depth_drop = True
                    elif self.best and child in self.frontier:
                        old_child = self.frontier[child]
                        if f(child) < f(old_child):
                            del self.frontier[old_child]
                            self.frontier.append(child)
                if self.local:
                    self.frontier.extend(self.local_q)
                    self.local_q.clear()
                printdb(0,"Frontier:",self.frontier)
        return None
    
    def bidirectional_search(self):
        while self.frontier and self.frontierB and self.num_expanded < self.max_expanded:
            if self.frontier:
                node = self.frontier.pop()
                printdb(0, ">>> Expanding:", node, "| cost =", node.path_cost)
                self.num_expanded += 1
                if not self.cycle or node.state not in node.solution()[:-1]:
                    to_return = None
                    if self.problem.is_goal(node.state) or node in self.frontierB:
                        printdb(0, ">>> ** Explored:", self.num_expanded, ",", len(self.frontier), "remained,", node, "-", self.frontierB[node], "found")
                        if self.problem.is_goal(node.state):
                            return node
                        else:
                            to_return = node.connect(self.problem, self.frontierB[node])
                            for sol in to_return.solution():
                                if to_return.solution().count(sol) > 1:
                                    to_return = None
                                    break
                    if self.graph:
                        self.explored.add(node.state)
                    for child in node.neighbors(self.problem):
                        if not self.graph or (child.state not in self.explored and child not in self.frontier):
                            if self.depth_limit < 0 or child.depth <= self.depth_limit:
                                if self.local:
                                    self.local_q.append(child)
                                else:
                                    self.frontier.append(child)
                            else:
                                self.depth_drop = True
                        elif self.best and child in self.frontier:
                            old_child = self.frontier[child]
                            if f(child) < f(old_child):
                                del self.frontier[old_child]
                                self.frontier.append(child)
                    if self.local:
                        self.frontier.extend(self.local_q)
                        self.local_q.clear()
                    printdb(0,">>> Frontier:",self.frontier)
                    if to_return:
                        return to_return
            if self.frontierB:
                node = self.frontierB.pop()
                printdb(0, "<<< Expanding:", node, "| cost =", node.path_cost)
                self.num_expanded += 1
                if not self.cycle or node.state not in node.solution()[:-1]:
                    to_return = None
                    if self.problem.is_start(node.state) or node in self.frontier:
                        printdb(0, "<<< ** Explored:", self.num_expanded, ",", len(self.frontier), "remained,", node,"-", self.frontier[node], "found")
                        if self.problem.is_start(node.state):
                            return node.reverse(self.problem)
                        else:
                            to_return = self.frontier[node].connect(self.problem, node)
                            for sol in to_return.solution():
                                if to_return.solution().count(sol) > 1:
                                    to_return = None
                                    break
                    if self.graph:
                        self.explored.add(node.state)
                    for child in node.neighbors_rev(self.problem):
                        if not self.graph or (child.state not in self.explored and child not in self.frontier):
                            if self.depth_limit < 0 or child.depth <= self.depth_limit:
                                if self.local:
                                    self.local_q.append(child)
                                else:
                                    self.frontierB.append(child)
                            else:
                                self.depth_drop = True
                        elif self.best and child in self.frontier:
                            old_child = self.frontier[child]
                            if f(child) < f(old_child):
                                del self.frontier[old_child]
                                self.frontier.append(child)
                    if self.local:
                        self.frontier.extend(self.local_q)
                        self.local_q.clear()
                    printdb(0,"<<< Frontier:",self.frontierB)
                    if to_return:
                        return to_return

