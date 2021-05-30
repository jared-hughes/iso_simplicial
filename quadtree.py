from typing import List, Callable
import numpy as np
import scipy.optimize

CYCLIC_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]
SHRINK_FACTOR = 0.05
# X_TOLERANCE and Y_TOLERANCE effectively determine the max depth
# TODO: compute X and Y tolerance based on x-width and y-width
#   divided by the screen size (pixels) to exclude details smaller than a pixel
X_TOLERANCE = 0.005
Y_TOLERANCE = 0.005
NORM_TOLERANCE = X_TOLERANCE ** 2 + Y_TOLERANCE ** 2
GRADIENT_EPS_X = X_TOLERANCE / 10
GRADIENT_EPS_Y = Y_TOLERANCE / 10
GRADIENT_EPS_MIN_SQ = min(GRADIENT_EPS_X, GRADIENT_EPS_Y) ** 2


def nonlinearity_along_edge(p1, p2, fn):
    midpoint = (p1 + p2) / 2
    # TODO: handle p1[2] == p2[2] == 0
    return np.abs(p1[2] - 2 * fn(midpoint[0], midpoint[1]) + p2[2]) / np.max(
        np.abs([p1[2], p2[2]])
    )


class Quadtree:
    def __init__(
        self,
        vertices: List[np.ndarray],
        depth: int,
        fn: Callable[[float, float], float],
    ):
        """Vertices are a list of (x, y, f(x,y)) values"""
        self.vertices = vertices
        self.depth = depth
        self.fn = fn
        self.children: List[Quadtree] = []

    def _apply_func_to(self, p: np.ndarray):
        """Mutates p"""
        p[2] = self.fn(p[0], p[1])
        return p

    def apply_func_to_vertices(self):
        for v in self.vertices:
            self._apply_func_to(v)

    def _construct_point(self, x: float, y: float):
        return np.array([x, y, self.fn(x, y)])

    def compute_midpoints(self):
        center = 0.5 * (self.vertices[0] + self.vertices[2])
        self.edge_midpoints = (
            self._construct_point(center[0], self.vertices[0][1]),
            self._construct_point(self.vertices[1][0], center[1]),
            self._construct_point(center[0], self.vertices[2][1]),
            self._construct_point(self.vertices[0][0], center[1]),
        )
        self.center_midpoint = self._construct_point(center[0], center[1])

    def compute_children(self):
        """Uniform subdivision"""
        self.compute_midpoints()
        next_depth = self.depth + 1
        self.children = [
            Quadtree(
                (
                    self.vertices[0],
                    self.edge_midpoints[0],
                    self.center_midpoint,
                    self.edge_midpoints[3],
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.edge_midpoints[0],
                    self.vertices[1],
                    self.edge_midpoints[1],
                    self.center_midpoint,
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.center_midpoint,
                    self.edge_midpoints[1],
                    self.vertices[2],
                    self.edge_midpoints[2],
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.edge_midpoints[3],
                    self.center_midpoint,
                    self.edge_midpoints[2],
                    self.vertices[3],
                ),
                next_depth,
                self.fn,
            ),
        ]

    def _get_edge_dual_between(self, i: int, j: int):
        """Solve exactly for the intersection between normal planes"""
        axis = i % 2

        p = self.vertices[i]
        q = self.vertices[j]
        m = self.vertex_gradients[i]
        n = self.vertex_gradients[j]
        dot = m[axis] * (q[axis] - p[axis]) + m[2] * (q[2] - p[2])
        det = m[axis] * n[2] - m[2] * n[axis]

        if np.abs(det) < GRADIENT_EPS_MIN_SQ:
            return self._apply_func_to(0.5 * (p + q))

        coord = q[axis] - m[2] * dot / det

        coord_min = (
            self.vertices[3][axis] * (1 - SHRINK_FACTOR)
            + self.vertices[1][axis] * SHRINK_FACTOR
        )
        coord_max = (
            self.vertices[1][axis] * (1 - SHRINK_FACTOR)
            + self.vertices[3][axis] * SHRINK_FACTOR
        )
        coord = np.clip(coord, coord_min, coord_max)

        if axis == 0:
            y = p[1]
            return np.array([coord, y, self.fn(coord, y)])
        else:
            x = p[0]
            return np.array([x, coord, self.fn(x, coord)])

    def _get_edge_duals(self):
        return [
            self._apply_func_to(self._get_edge_dual_between(i, j))
            for i, j in CYCLIC_PAIRS
        ]

    def _get_face_dual(self):
        # assume self.vertex_gradients is already computed
        B = [np.dot(n, p) for n, p in zip(self.vertex_gradients, self.vertices)]

        x_min, y_min, _ = (
            self.vertices[3] * (1 - SHRINK_FACTOR) + SHRINK_FACTOR * self.vertices[1]
        )
        x_max, y_max, _ = (
            self.vertices[3] * SHRINK_FACTOR + (1 - SHRINK_FACTOR) * self.vertices[1]
        )
        result = scipy.optimize.lsq_linear(
            self.vertex_gradients,
            B,
            bounds=([x_min, y_min, -np.inf], [x_max, y_max, np.inf]),
            method="bvls",
            # if within `distance` of the correct minimum,
            # then the cost function is at least `distance**2`
            tol=NORM_TOLERANCE,
        )
        return self._apply_func_to(result.x)

    def _compute_gradients(self):
        # gradients of G = f(x,y)-z
        # we compute these numerically based on a finite difference
        # TODO: compute these in a parent quad and pass it down
        # to reduce repeated computation
        self.vertex_gradients = [
            np.array(
                [
                    (self.fn(v[0] + GRADIENT_EPS_X, v[1]) - v[2]) / GRADIENT_EPS_X,
                    (self.fn(v[0], v[1] + GRADIENT_EPS_Y) - v[2]) / GRADIENT_EPS_Y,
                    -1,
                ]
            )
            for v in self.vertices
        ]

    def compute_duals(self):
        """Insert dual vertices. In the future, this should only be done for minimal cells"""
        self._compute_gradients()
        self.edge_duals = self._get_edge_duals()
        self.face_dual = self._get_face_dual()
        self.nonlinearity = np.max(
            [
                nonlinearity_along_edge(self.vertices[i], self.vertices[j], self.fn)
                for i, j in CYCLIC_PAIRS + [(0, 2), (1, 3)]
            ]
        )

    """ Following are debug functions """

    def _directional_duals(self, direction: int):
        """All edge duals, including children, in clockwise order as an iterator
        of tuples (xy position, value)
        Direction: 0=top, 1=right, 2=bottom, 3=left"""
        if len(self.children) != 0:
            yield from self.children[direction]._directional_duals(direction)
        yield self.edge_duals[direction]
        if len(self.children) != 0:
            yield from self.children[(direction + 1) % 4]._directional_duals(direction)

    def all_duals(self):
        """All edge duals and vertices, including children, in clockwise order"""
        for i in range(4):
            yield self.vertices[i]
            yield from self._directional_duals(i)

    def leaves(self):
        if len(self.children) == 0:
            yield self
        for child in self.children:
            yield from child.leaves()

    def __str__(self):
        return f"Quadtree[depth={self.depth}, {self.left}≤x≤{self.right}, {self.bottom}≤y≤{self.top}, children:{len(self.children)}]"

    def visualize_borders(self):
        """Returns a list of points [x: str | float, y: str | float] which draw the quadtree when connected in Desmos"""
        if len(self.children) > 0:
            yield self.edge_midpoints[0]
            yield self.edge_midpoints[2]
            yield ["0/0", "0"]
            yield self.edge_midpoints[3]
            yield self.edge_midpoints[1]
            yield ["0/0", "0"]
        for child in self.children:
            yield from child.visualize_borders()

    def leaf_all_duals(self):
        if len(self.children) == 0:
            yield from self.all_duals()
        for child in self.children:
            yield from child.leaf_all_duals()

    def leaf_face_duals(self):
        if len(self.children) == 0:
            yield self.face_dual
        for child in self.children:
            yield from child.leaf_face_duals()

    @property
    def width(self):
        return self.vertices[1][0] - self.vertices[0][0]

    @property
    def height(self):
        return self.vertices[1][1] - self.vertices[2][1]
