""" utilities for use """
import functools
import math
import collections
import bisect


def forEach(f, seq):
    for elt in seq:
        f(elt)

def is_in(elt, seq):
    """ Use is for comparison in list"""
    return any(x is elt for x in seq)

def printdb(lvl, *args,**nargs):
    if lvl:
        print(*args, **nargs)


class Queue():

    """Queue is an abstract class/interface. There are three types:
        Stack():                    A Last In First Out Queue.
        FIFOQueue():                A First In First Out Queue.
        PriorityQueue(order, f):    Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
        for item in q   -- iterator
    """

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)

class Stack(Queue):
    def __init__(self):
        self.queue = []
    
    def append(self, item):
        self.queue.append(item)
    
    def extend(self, items):
        self.queue.extend(items)
    
    def clear(self):
        del self.queue[:]
    
    def pop(self):
        return self.queue.pop()
    
    def __len__(self):
        return len(self.queue)
    
    def __contains__(self, item):
        return item in self.queue
    
    def __getitem__(self, key):
        for item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        self.remove(key)
    
    def __repr__(self):
        return str(self.queue)
    
    def __iter__(self):
        return iter(self.queue)


class FIFOQueue(Queue):

    """A First-In-First-Out Queue."""

    def __init__(self, maxlen=None, items=[]):
        self.queue = collections.deque(items, maxlen)

    def append(self, item):
        if not self.queue.maxlen or len(self.queue) < self.queue.maxlen:
            self.queue.append(item)
        else:
            raise Exception('FIFOQueue is full')

    def extend(self, items):
        if not self.queue.maxlen or len(self.queue) + len(items) <= self.queue.maxlen:
            self.queue.extend(items)
        else:
            raise Exception('FIFOQueue max length exceeded')
    
    def clear(self):
        while len(self.queue) > 0:
            self.queue.popleft()

    def pop(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            raise Exception('FIFOQueue is empty')

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return item in self.queue
        
    def __getitem__(self, key):
        for item in self.queue:
            if item == key:
                return item

    def __delitem__(self, key):
        self.remove(key)
    
    def __repr__(self):
        return str(self.queue)
    
    def __iter__(self):
        return iter(self.queue)


class PriorityQueue(Queue):

    """A pqueue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with min f(x) is
    returned first; if order is max, then it is the item with max f(x).
    """

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def clear(self):
        while len(self.A) > 0:
            self.A.pop()
    
    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
    
    def __repr__(self):
        return str([(value, str(item)) for (value, item) in self.A])
    
    def __iter__(self):
        if self.order == min:
            return iter(map(lambda item: item[1], self.A))
        else:
            return iter(reversed(map(lambda item: item[1], self.A)))


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return math.hypot((xA - xB), (yA - yB))