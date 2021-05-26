from typing import List
from dataclasses import dataclass


@dataclass
class Rect:
    """+x to the right, +y to the top, so right>left and top>bottom"""

    left: float
    right: float
    bottom: float
    top: float

    def __str__(self):
        return f"Rect[{self.left} ≤ x ≤ {self.right}, {self.bottom} ≤ y ≤ {self.top}]"


class Quadtree:
    def __init__(self, boundary: Rect, depth: int):
        self.boundary = boundary
        self.children: List[Quadtree] = []
        self.depth = depth

    def subdivide(self):
        """Uniform subdivision"""
        mx = (self.boundary.left + self.boundary.right) / 2
        my = (self.boundary.top + self.boundary.bottom) / 2
        nw = Rect(self.boundary.left, mx, my, self.boundary.top)
        ne = Rect(mx, self.boundary.right, my, self.boundary.top)
        se = Rect(mx, self.boundary.right, self.boundary.bottom, my)
        sw = Rect(self.boundary.left, mx, self.boundary.bottom, my)
        next_depth = self.depth + 1
        self.children = [
            Quadtree(nw, next_depth),
            Quadtree(ne, next_depth),
            Quadtree(se, next_depth),
            Quadtree(sw, next_depth),
        ]

    def subdivide_to_depth(self, depth: int):
        """Perform uniform subdivision until the quadtree reaches a depth of `depth`"""
        if depth > self.depth:
            self.subdivide()
            for quad in self.children:
                quad.subdivide_to_depth(depth)
