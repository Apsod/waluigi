from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from unittest.mock import sentinel


import concurrent.futures as futures
import asyncio

import copy

@dataclass(frozen=True, eq=True)
class Directed:
    val: Any
    
    @classmethod
    def opposite(cls, val):
        pass

    def flipped(self):
        return self.opposite(self.val)

@dataclass(frozen=True, eq=True)
class Left(Directed):
    @classmethod
    def opposite(cls, val):
        return Right(val)

@dataclass(frozen=True, eq=True)
class Right(Directed):
    @classmethod
    def opposite(cls, val):
        return Left(val)

@dataclass(frozen=True, eq=True)
class NodeInfo:
    left: {Any}
    right: {Any}

class Graph(object):
    def __init__(self):
        self.edges = defaultdict(set)
        self.leftmost = Left(sentinel.leftmost)
        self.rightmost = Right(sentinel.rightmost)
        self._add(self.leftmost, self.rightmost)
    
    def _add(self, L, R):
        assert isinstance(L, Left), f'{L} is not Left'
        assert isinstance(R, Right), f'{R} is not Right'
        self.edges[L].add(R.val)
        self.edges[R].add(L.val)

    def _remove(self, L, R):
        assert isinstance(L, Left), f'{L} is not Left'
        assert isinstance(R, Right), f'{R} is not Right'
        self.edges[L].discard(R.val)
        self.edges[R].discard(L.val)

        if not self.edges[L]:
            del self.edges[L]
        if not self.edges[R]:
            del self.edges[R]

    def add(self, l, r=None):
        if r is None and not self.has(l):
            self._add(self.leftmost, Right(l))
            self._add(Left(l), self.rightmost)
        elif r is not None:
            L = Left(l)
            R = Right(r)

            self._add(L, R)

            # Add anchors
            if L.flipped() not in self.edges:
                self._add(self.leftmost, L.flipped())
            if R.flipped() not in self.edges:
                self._add(R.flipped(), self.rightmost)
            # Discard anchors
            self._remove(self.leftmost, R)
            self._remove(L, self.rightmost)

    def get(self, node):
        if Right(node) in self.edges:
            left = {x for x in self.edges[Right(node)] if x != self.leftmost.val}
        else:
            left = {}
        if Left(node) in self.edges:
            right = {x for x in self.edges[Left(node)] if x != self.rightmost.val}
        else:
            right = {}
        return NodeInfo(
                left = left,
                right = right,
                )

    def empty(self):
        return not bool(self.edges)

    def has(self, x, which=Left):
        return which(x) in self.edges

    def pop(self, x):
        """
        Pop an edge from x
        """
        if x not in self.edges:
            raise KeyError

        y = x.opposite(self.edges[x].pop())
        assert x.val in self.edges[y]
        self.edges[y].discard(x.val)

        if not self.edges[y]: # if child has no parents: remove it
            del self.edges[y]
        if not self.edges[x]: # if parent has no children: remove it
            del self.edges[x]
        return y.val
    
    def pops(self, x):
        try:
            while True:
                yield self.pop(x)
        except KeyError:
            pass

    def copy(self):
        ret = Graph()
        ret.leftmost = self.leftmost
        ret.rightmost = self.rightmost
        for l, r in self.edges.items():
            ret.edges[l] = copy.copy(r)
        return ret

    def root(self, direction=Left):
        return self.leftmost.val if direction is Left else self.rightmost.val

    def kahns(self, direction=Left, pure=True):
        gen = self._kahns(direction, pure)
        roots = next(gen)
        while roots:
            n = roots.pop()
            roots.update(gen.send(n))
            yield n

    def _kahns(self, direction=Left, pure=True):
        graph = self.copy() if pure else self
        roots = {self.leftmost.val} if direction is Left else {self.rightmost.val}
        n = yield roots
        while True:
            new_roots = set()
            for child in graph.pops(direction(n)):
                if not direction.opposite(child) in graph.edges:
                    new_roots.add(child)
            n = yield new_roots

    def toposort(self, pure=True):
        tmp = self.copy() if pure else self
        sorted = list(tmp.kahns(pure=False))
        if not tmp.empty():
            lines = ['']
            for l, rs in tmp.edges.items():
                lines.append(f'{l.val}')
                for r in rs:
                    lines.append(f'\t=> {r}')
            lines = '\n'.join(lines)
            raise ValueError(f'Cycles detected: {lines}')
        return sorted

    def print(self):
        sorted = self.toposort()
        for node in sorted:
            info = self.get(node)
            print(node)
            for l in info.left:
                print(f'\t<= {l}')
            for r in info.right:
                print(f'\t=> {r}')
