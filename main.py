from generic import *
from tabulate import tabulate

f = open('output.txt', 'w')

table = []
for method in ['Depth-first', 'Breadth-first', 'Lowest-cost-first', 'Depth-limited', 'Iterative deepening', 'Heuristic depth-first', 'Greedy best-first', 'A*', 'Bidirectional']:
    if method == 'Depth-limited':
        s = Searcher(cyclic_delivery_problem, method, 6)
    elif method == 'Bidirectional':
        s = Searcher(cyclic_delivery_problem, 'Breadth-first', bidir = True)
    else:
        s = Searcher(cyclic_delivery_problem, method)
    table += [[method, str(s.search()), "{{{nodes}}}".format(nodes = "\n ".join(str(node) for node in s.search_all()))]]



"""
Write to file
"""


f.write(tabulate(table, headers = ['Search Strategy', 'First Solution Path', 'Set of All Solution Paths'], tablefmt="pipe"))
f.close()

