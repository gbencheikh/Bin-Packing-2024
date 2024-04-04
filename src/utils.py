import itertools
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from collections import defaultdict
from loguru import logger

from rectpack import newPacker, PackingMode, PackingBin, SORT_AREA
from rectpack.maxrects import MaxRectsBaf, MaxRectsBl, MaxRectsBlsf, MaxRectsBssf


class Dimension:
    """
    Helper class to define object dimensions
    """

    def __init__(self, width, depth, height, weight=0):
        self.width = int(width)
        self.depth = int(depth)
        self.height = int(height)
        self.weight = int(weight)
        self.area = int(width * depth)
        self.volume = int(width * depth * height)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.width == other.width
                and self.depth == other.depth
                and self.height == other.height
                and self.weight == other.weight
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Dimension(width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Coordinate:
    """
    Helper class to define a pair/triplet of coordinates
    (defined as the bottom-left-back point of a cuboid)
    """

    def __init__(self, x, y, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def from_blb_to_vertices(self, dims):
        """
        Convert bottom-left-back coordinates to
        the list of all vertices in the cuboid
        """
        assert isinstance(dims, Dimension), "The given dimension should be an instance of Dimension"
        blb = self
        blf = Coordinate(self.x + dims.width, self.y, self.z)
        brb = Coordinate(self.x, self.y + dims.depth, self.z)
        brf = Coordinate(self.x + dims.width, self.y + dims.depth, self.z)
        tlb = Coordinate(self.x, self.y, self.z + dims.height)
        tlf = Coordinate(self.x + dims.width, self.y, self.z + dims.height)
        trb = Coordinate(self.x, self.y + dims.depth, self.z + dims.height)
        trf = Coordinate(self.x + dims.width, self.y + dims.depth, self.z + dims.height)
        return [blb, blf, brb, brf, tlb, tlf, trb, trf]

    def to_numpy(self):
        """
        Convert coordinates to a numpy list
        """
        return np.array([self.x, self.y, self.z])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))


class Vertices:
    """
    Helper class to define the set of vertices identifying a cuboid
    """

    def __init__(self, blb, dims):
        assert isinstance(
            blb, Coordinate
        ), "The given coordinate should be an instance of Coordinate"
        assert isinstance(dims, Dimension), "The given dimension should be an instance of Dimension"
        self.dims = dims

        # Bottom left back and front
        self.blb = blb
        self.blf = Coordinate(self.blb.x + self.dims.width, self.blb.y, self.blb.z)

        # Bottom right back and front
        self.brb = Coordinate(self.blb.x, self.blb.y + self.dims.depth, self.blb.z)
        self.brf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y + self.dims.depth, self.blb.z
        )

        # Top left back and front
        self.tlb = Coordinate(self.blb.x, self.blb.y, self.blb.z + self.dims.height)
        self.tlf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y, self.blb.z + self.dims.height
        )

        # Top right back and front
        self.trb = Coordinate(
            self.blb.x, self.blb.y + self.dims.depth, self.blb.z + self.dims.height
        )
        self.trf = Coordinate(
            self.blb.x + self.dims.width,
            self.blb.y + self.dims.depth,
            self.blb.z + self.dims.height,
        )

        # List of vertices
        self.vertices = [
            self.blb,
            self.blf,
            self.brb,
            self.brf,
            self.tlb,
            self.tlf,
            self.trb,
            self.trf,
        ]

    def get_center(self):
        """
        Return the central coordinate of the cuboid
        """
        return Coordinate(
            self.blb.x + self.dims.width // 2,
            self.blb.y + self.dims.depth // 2,
            self.blb.z + self.dims.height // 2,
        )

    def get_xs(self):
        """
        Return a numpy array containing all the x-values
        of the computed vertices
        """
        return np.array([v.x for v in self.vertices])

    def get_ys(self):
        """
        Return a numpy array containing all the y-values
        of the computed vertices
        """
        return np.array([v.y for v in self.vertices])

    def get_zs(self):
        """
        Return a numpy array containing all the z-values
        of the computed vertices
        """
        return np.array([v.z for v in self.vertices])

    def to_faces(self):
        """
        Convert the computed set of vertices to a list of faces
        (6 different faces for one cuboid)
        """
        return np.array(
            [
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.brb.to_numpy(),
                ],  # bottom
                [
                    self.tlb.to_numpy(),
                    self.tlf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # top
                [
                    self.blb.to_numpy(),
                    self.brb.to_numpy(),
                    self.trb.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # back
                [
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.tlf.to_numpy(),
                ],  # front
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.tlf.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # left
                [
                    self.brb.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # right
            ]
        )


def argsort(seq, reverse=False):
    """
    Sort the given array and return indices instead of values
    """
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def duplicate_keys(dicts):
    """
    Check that the input dictionaries have common keys
    """
    keys = list(flatten([d.keys() for d in dicts]))
    return [k for k, v in Counter(keys).items() if v > 1]


def flatten(l):
    """
    Given nested Python lists, return their flattened version
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def build_layer_from_model_output(superitems_pool, superitems_in_layer, solution, pallet_dims):
    """
    Return a single layer from the given model solution (either baseline or column generation).
    The 'solution' parameter should be a dictionary of the form
    {
        'c_{s}_x': ...,
        'c_{s}_y: ...,
        ...
    }
    """
    spool, scoords = [], []
    for s in superitems_in_layer:
        spool += [superitems_pool[s]]
        scoords += [Coordinate(x=solution[f"c_{s}_x"], y=solution[f"c_{s}_y"])]
    spool = SuperitemPool(superitems=spool)
    return Layer(spool, scoords, pallet_dims)


def do_overlap(a, b):
    """
    Check if the given items strictly overlap or not
    (both items should be given as a Pandas Series)
    """
    assert isinstance(a, pd.Series) and isinstance(b, pd.Series), "Wrong input types"
    dx = min(a.x.item() + a.width.item(), b.x.item() + b.width.item()) - max(a.x.item(), b.x.item())
    dy = min(a.y.item() + a.depth.item(), b.y.item() + b.depth.item()) - max(a.y.item(), b.y.item())
    if (dx > 0) and (dy > 0):
        return True
    return False


def get_pallet_plot(pallet_dims):
    """
    Compute an initial empty 3D-plot with the pallet dimensions
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("xkcd:white")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.text(0, 0, 0, "origin", size=10, zorder=1, color="k")
    ax.view_init(azim=60)
    ax.set_xlim3d(0, pallet_dims.width)
    ax.set_ylim3d(0, pallet_dims.depth)
    ax.set_zlim3d(0, pallet_dims.height)
    return ax


def plot_product(ax, item_id, coords, dims):
    """
    Add product to given axis
    """
    vertices = Vertices(coords, dims)
    ax.scatter3D(vertices.get_xs(), vertices.get_ys(), vertices.get_zs())
    ax.add_collection3d(
        Poly3DCollection(
            vertices.to_faces(),
            facecolors=np.random.rand(1, 3),
            linewidths=1,
            edgecolors="r",
            alpha=0.45,
        )
    )
    center = vertices.get_center()
    ax.text(
        center.x,
        center.y,
        center.z,
        item_id,
        size=10,
        zorder=1,
        color="k",
    )
    return ax


def get_l0_lb(order, pallet_dims):
    """
    L0 lower bound (aka continuos lower bound) for the
    minimum number of required bins. The worst case
    performance of this bound is 1 / 8.

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """
    return np.ceil(order.volume.sum() / pallet_dims.volume)


def get_l1_lb(order, pallet_dims):
    """
    L1 lower bound for the minimum number of required bins.
    The worst-case performance of L1 can be arbitrarily bad.

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """

    def get_j2(d1, bd1, d2, bd2):
        return order[(order[d1] > (bd1 / 2)) & (order[d2] > (bd2 / 2))]

    def get_js(j2, p, d, bd):
        return j2[(j2[d] >= p) & (j2[d] <= (bd / 2))]

    def get_jl(j2, p, d, bd):
        return j2[(j2[d] > (bd / 2)) & (j2[d] <= bd - p)]

    def get_l1j2(d1, bd1, d2, bd2, d3, bd3):
        j2 = get_j2(d1, bd1, d2, bd2)
        if len(j2) == 0:
            return 0.0
        ps = order[order[d3] <= bd3 / 2][d3].values
        max_ab = -np.inf
        for p in tqdm(ps):
            js = get_js(j2, p, d3, bd3)
            jl = get_jl(j2, p, d3, bd3)
            a = np.ceil((js[d3].sum() - (len(jl) * bd3 - jl[d3].sum())) / bd3)
            b = np.ceil((len(js) - (np.floor((bd3 - jl[d3].values) / p)).sum()) / np.floor(bd3 / p))
            max_ab = max(max_ab, a, b)

        return len(j2[j2[d3] > (bd3 / 2)]) + max_ab

    l1wh = get_l1j2(
        "width", pallet_dims.width, "height", pallet_dims.height, "depth", pallet_dims.depth
    )
    l1wd = get_l1j2(
        "width", pallet_dims.width, "depth", pallet_dims.depth, "height", pallet_dims.height
    )
    l1dh = get_l1j2(
        "depth", pallet_dims.depth, "width", pallet_dims.width, "height", pallet_dims.height
    )
    return max(l1wh, l1wd, l1dh), l1wh, l1wd, l1dh


def get_l2_lb(order, pallet_dims):
    """
    L2 lower bound for the minimum number of required bins
    The worst-case performance of this bound is 2 / 3.

    Silvano Martello, David Pisinger and Daniele Vigo,
    "The Three-Dimensional Bin Packing Problem",
    Operations Research, 1998.
    """

    def get_kv(p, q, d1, bd1, d2, bd2):
        return order[(order[d1] > bd1 - p) & (order[d2] > bd2 - q)]

    def get_kl(kv, d1, bd1, d2, bd2):
        kl = order[~order.isin(kv)]
        return kl[(kl[d1] > (bd1 / 2)) & (kl[d2] > (bd2 / 2))]

    def get_ks(kv, kl, p, q, d1, d2):
        ks = order[~order.isin(pd.concat([kv, kl], axis=0))]
        return ks[(ks[d1] >= p) & (ks[d2] >= q)]

    def get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3):
        kv = get_kv(p, q, d1, bd1, d2, bd2)
        kl = get_kl(kv, d1, bd1, d2, bd2)
        ks = get_ks(kv, kl, p, q, d1, d2)

        return l1 + max(
            0,
            np.ceil(
                (pd.concat([kl, ks], axis=0).volume.sum() - (bd3 * l1 - kv[d3].sum()) * bd1 * bd2)
                / (bd1 * bd2 * bd3)
            ),
        )

    def get_l2j2(l1, d1, bd1, d2, bd2, d3, bd3):
        ps = order[(order[d1] <= bd1 // 2)][d1].values
        qs = order[(order[d2] <= bd2 // 2)][d2].values
        max_l2j2 = -np.inf
        for p, q in tqdm(itertools.product(ps, qs)):
            l2j2 = get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3)
            max_l2j2 = max(max_l2j2, l2j2)
        return max_l2j2

    _, l1wh, l1wd, l1hd = get_l1_lb(order, pallet_dims)
    l2wh = get_l2j2(
        l1wh, "width", pallet_dims.width, "height", pallet_dims.height, "depth", pallet_dims.depth
    )
    l2wd = get_l2j2(
        l1wd, "width", pallet_dims.width, "depth", pallet_dims.depth, "height", pallet_dims.height
    )
    l2dh = get_l2j2(
        l1hd, "depth", pallet_dims.depth, "height", pallet_dims.height, "width", pallet_dims.width
    )
    return max(l2wh, l2wd, l2dh), l2wh, l2wd, l2dh


class Item:
    """
    An item is a single product with a unique identifier
    and a list of spatial dimensions
    """

    def __init__(self, id, width, depth, height, weight):
        self.id = id
        self.dimensions = Dimension(width, depth, height, weight)

    @classmethod
    def from_series(cls, item):
        """
        Return an item from a Pandas Series representing a row of
        the order extracted from the ProductDataset custom class
        """
        return Item(item.name, item.width, item.depth, item.height, item.weight)

    @classmethod
    def from_dataframe(cls, order):
        """
        Return a list of items from a Pandas DataFrame obtained
        as an order from the ProductDataset custom class
        """
        return [Item(i.name, i.width, i.depth, i.height, i.weight) for _, i in order.iterrows()]

    @property
    def width(self):
        """
        Return the width of the item
        """
        return self.dimensions.width

    @property
    def depth(self):
        """
        Return the depth of the item
        """
        return self.dimensions.depth

    @property
    def height(self):
        """
        Return the height of the item
        """
        return self.dimensions.height

    @property
    def weight(self):
        """
        Return the weight of the item
        """
        return self.dimensions.weight

    @property
    def volume(self):
        """
        Return the volume of the item
        """
        return self.dimensions.volume

    @property
    def area(self):
        """
        Return the area of the item
        """
        return self.dimensions.area

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id and self.dimensions == other.dimensions
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Item(id={self.id}, width={self.width}, depth={self.depth}, "
            f"height={self.height}, weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Superitem:
    """
    A superitem is a grouping of items or superitems
    having almost the same dimensions
    """

    def __init__(self, items):
        # Represents a list of superitems
        self.items = items

    @property
    def width(self):
        """
        Return the width of the superitem
        """
        raise NotImplementedError()

    @property
    def depth(self):
        """
        Return the depth of the superitem
        """
        raise NotImplementedError()

    @property
    def height(self):
        """
        Return the height of the superitem
        """
        raise NotImplementedError()

    @property
    def enclosing_volume(self):
        """
        Return the volume of the minimum sized rectangle
        fully enclosing the superitem
        """
        raise NotImplementedError()

    @property
    def weight(self):
        """
        Compute the weight of the superitem as the
        sum of the item weights it's composed of
        """
        return sum(i.weight for i in self.items)

    @property
    def volume(self):
        """
        Compute the volume of the superitem as the
        sum of the item volumes it's composed of
        """
        return sum(i.volume for i in self.items)

    @property
    def area(self):
        """
        Compute the area of the superitem as the
        sum of the item areas it's composed of
        """
        return sum(i.area for i in self.items)

    @property
    def id(self):
        """
        Return a sorted list of item ids contained in the superitem
        """
        return sorted(flatten([i.id for i in self.items]))

    def get_items(self):
        """
        Return a list of single items in the superitem
        """
        return list(flatten([i.items for i in self.items]))

    def get_num_items(self):
        """
        Return the number of single items in the superitem
        """
        return len(self.id)

    def get_items_coords(self, width=0, depth=0, height=0):
        """
        Return a dictionary c of coordinates with one entry for each
        item in the superitem, s.t. c[i] = (x, y, z) represents the
        coordinates of item i relative to the superitem itself
        """
        raise NotImplementedError()

    def get_items_dims(self):
        """
        Return a dictionary d of dimensions with one entry for each
        item in the superitem, s.t. d[i] = (w, d, h) represents the
        dimensions of item i in the superitem
        """
        all_dims = dict()
        for i in range(len(self.items)):
            dims = self.items[i].get_items_dims()
            dups = duplicate_keys([all_dims, dims])
            assert len(dups) == 0, f"Duplicated item in the same superitem, item ids: {dups}"
            all_dims = {**all_dims, **dims}
        return all_dims

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.id == other.id
                and self.width == other.width
                and self.depth == other.depth
                and self.height == other.height
                and self.weight == other.weight
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Superitem(ids={self.id}, width={self.width}, depth={self.depth}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume}, coords={self.get_items_coords()})"
        )

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum(hash(str(i)) for i in self.id)


class SingleItemSuperitem(Superitem):
    """
    Superitem containing a single item
    """

    def __init__(self, items):
        assert len(items) == 1
        super().__init__(items)

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

    @property
    def height(self):
        return max(i.height for i in self.items)

    @property
    def enclosing_volume(self):
        return self.volume

    def get_items_coords(self, width=0, depth=0, height=0):
        return {self.items[0].id: Coordinate(width, depth, height)}

    def get_items_dims(self):
        return {self.items[0].id: self.items[0].dimensions}


class HorizontalSuperitem(Superitem):
    """
    An horizontal superitem is a group of 2 or 4 items (not superitems)
    that have exactly the same dimensions and get stacked next to each other
    """

    def __init__(self, items):
        super().__init__(items)

    @property
    def height(self):
        return max(i.height for i in self.items)

    @property
    def enclosing_volume(self):
        return self.volume


class TwoHorizontalSuperitemWidth(HorizontalSuperitem):
    """
    Horizontal superitem with 2 items stacked by the width dimension
    """

    def __init__(self, items):
        assert len(items) == 2
        super().__init__(items)

    @property
    def width(self):
        return sum(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=width + i1.width, depth=depth, height=height)
        dups = duplicate_keys([d1, d2])
        assert len(dups) == 0, f"Duplicated item in the same superitem, item ids: {dups}"
        return {**d1, **d2}


class TwoHorizontalSuperitemDepth(HorizontalSuperitem):
    """
    Horizontal superitem with 2 items stacked by the depth dimension
    """

    def __init__(self, items):
        assert len(items) == 2
        super().__init__(items)

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return sum(i.depth for i in self.items)

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=width, depth=i1.depth + depth, height=height)
        dups = duplicate_keys([d1, d2])
        assert len(dups) == 0, f"Duplicated item in the same superitem, items ids: {dups}"
        return {**d1, **d2}


class FourHorizontalSuperitem(HorizontalSuperitem):
    """
    Horizontal superitem with 4 items stacked by the width and depth dimensions
    """

    def __init__(self, items):
        assert len(items) == 4
        super().__init__(items)

    @property
    def width(self):
        return sum(i.width for i in self.items)

    @property
    def depth(self):
        return sum(i.depth for i in self.items)

    def get_items_coords(self, width=0, depth=0, height=0):
        i1, i2, i3, i4 = tuple(self.items)
        d1 = i1.get_items_coords(width=width, depth=depth, height=height)
        d2 = i2.get_items_coords(width=i1.width + width, depth=depth, height=height)
        d3 = i3.get_items_coords(width=width, depth=i1.depth + depth, height=height)
        d4 = i4.get_items_coords(width=i1.width + width, depth=i1.depth + depth, height=height)
        dups = duplicate_keys([d1, d2, d3, d4])
        assert len(dups) == 0, f"Duplicated item in the same superitem, item ids: {dups}"
        return {**d1, **d2, **d3, **d4}


class VerticalSuperitem(Superitem):
    """
    A vertical superitem is a group of >= 2 items or horizontal superitems
    that have similar dimensions and get stacked on top of each other
    """

    def __init__(self, items):
        super().__init__(items)

    @property
    def width(self):
        return max(i.width for i in self.items)

    @property
    def depth(self):
        return max(i.depth for i in self.items)

    @property
    def height(self):
        return sum(i.height for i in self.items)

    @property
    def area(self):
        return self.width * self.depth

    @property
    def enclosing_volume(self):
        return self.width * self.depth * self.height

    def get_items_coords(self, width=0, depth=0, height=0):
        # Adjust coordinates to account for stacking tolerance
        all_coords = dict()
        for i in range(len(self.items)):
            width_offset = ((self.width - self.items[i].width) // 2) + width
            depth_offset = ((self.depth - self.items[i].depth) // 2) + depth
            coords = self.items[i].get_items_coords(
                width=width_offset,
                depth=depth_offset,
                height=height,
            )
            dups = duplicate_keys([all_coords, coords])
            assert len(dups) == 0, f"Duplicated item in the same superitem, item ids: {dups}"
            all_coords = {**all_coords, **coords}
            height += self.items[i].height

        return all_coords


class SuperitemPool:
    """
    Set of superitems for a given order
    """

    def __init__(self, superitems=None):
        self.superitems = superitems or []
        self.hash_to_index = self._get_hash_to_index()

    def _get_hash_to_index(self):
        """
        Compute a mapping for all superitems in the pool, with key
        the hash of the superitem and value its index in the pool
        """
        return {hash(s): i for i, s in enumerate(self.superitems)}

    def subset(self, superitems_indices):
        """
        Return a new superitems pool with the given subset of superitems
        """
        superitems = [s for i, s in enumerate(self.superitems) if i in superitems_indices]
        return SuperitemPool(superitems=superitems)

    def difference(self, superitems_indices):
        """
        Return a new superitems pool without the given subset of superitems
        """
        superitems = [s for i, s in enumerate(self.superitems) if i not in superitems_indices]
        return SuperitemPool(superitems=superitems)

    def add(self, superitem):
        """
        Add the given Superitem to the current SuperitemPool
        """
        assert isinstance(
            superitem, Superitem
        ), "The given superitem should be an instance of the Superitem class"
        s_hash = hash(superitem)
        if s_hash not in self.hash_to_index:
            self.superitems.append(superitem)
            self.hash_to_index[s_hash] = len(self.superitems) - 1

    def extend(self, superitems_pool):
        """
        Extend the current pool with the given one
        """
        assert isinstance(superitems_pool, SuperitemPool) or isinstance(
            superitems_pool, list
        ), "The given set of superitems should be an instance of the SuperitemPool class or a list"
        for superitem in superitems_pool:
            self.add(superitem)

    def remove(self, superitem):
        """
        Remove the given superitem from the pool
        """
        assert isinstance(
            superitem, Superitem
        ), "The given superitem should be an instance of the Superitem class"
        s_hash = hash(superitem)
        if s_hash in self.hash_to_index:
            del self.superitems[self.hash_to_index[s_hash]]
            self.hash_to_index = self._get_hash_to_index()

    def pop(self, i):
        """
        Remove the superitem at the given index from the pool
        """
        self.remove(self.superitems[i])

    def get_fsi(self):
        """
        Return a binary matrix of superitems by items, s.t.
        fsi[s, i] = 1 iff superitems s contains item i
        """
        item_ids = sorted(self.get_unique_item_ids())
        indexes = list(range(len(item_ids)))
        from_index_to_item_id = dict(zip(indexes, item_ids))
        from_item_id_to_index = dict(zip(item_ids, indexes))

        fsi = np.zeros((len(self.superitems), self.get_num_unique_items()), dtype=np.int32)
        for s, superitem in enumerate(self):
            for item_id in superitem.id:
                fsi[s, from_item_id_to_index[item_id]] = 1

        return fsi, from_index_to_item_id, from_item_id_to_index

    def get_superitems_dims(self):
        """
        Return the dimensions of each superitem in the pool
        """
        ws = [s.width for s in self.superitems]
        ds = [s.depth for s in self.superitems]
        hs = [s.height for s in self.superitems]
        return ws, ds, hs

    def get_superitems_containing_item(self, item_id):
        """
        Return a list of superitems containing the given item id
        """
        superitems, indices = [], []
        for i, superitem in enumerate(self.superitems):
            if item_id in superitem.id:
                superitems += [superitem]
                indices += [i]
        return superitems, indices

    def get_single_superitems(self):
        """
        Return the list of single item superitems in the pool
        """
        singles = []
        for superitem in self.superitems:
            if isinstance(superitem, SingleItemSuperitem):
                singles += [superitem]
        return singles

    def get_extreme_superitem(self, minimum=False, two_dims=False):
        """
        Return the superitem with minimum (or maximum) area
        (or volume) in the pool, along with its index
        """
        func = np.argmax if not minimum else np.argmin
        index = (
            func([s.area for s in self.superitems])
            if two_dims
            else func([s.volume for s in self.superitems])
        )
        return self.superitems[index], index

    def get_item_ids(self):
        """
        Return the ids of each superitem inside the pool, where each
        id is a list made up of the item ids contained in the superitem
        """
        return [s.id for s in self.superitems]

    def get_unique_item_ids(self):
        """
        Return the flattened list of ids of each item inside the pool
        """
        return sorted(set(flatten(self.get_item_ids())))

    def get_num_unique_items(self):
        """
        Return the total number of unique items inside the pool
        """
        return len(self.get_unique_item_ids())

    def get_volume(self):
        """
        Return the sum of superitems' volumes in the pool
        """
        return sum(s.volume for s in self.superitems)

    def get_max_height(self):
        """
        Return the maximum height of the superitems in the pool
        """
        if len(self.superitems) == 0:
            return 0
        return max(s.height for s in self.superitems)

    def get_index(self, superitem):
        """
        Return the index of the given superitem in the pool,
        if present, otherwise return None
        """
        assert isinstance(
            superitem, Superitem
        ), "The given superitem must be an instance of the Superitem class"
        return self.hash_to_index.get(hash(superitem))

    def to_dataframe(self):
        """
        Convert the pool to a DataFrame instance
        """
        ws, ds, hs = self.get_superitems_dims()
        ids = self.get_item_ids()
        types = [s.__class__.__name__ for s in self.superitems]
        return pd.DataFrame({"width": ws, "depth": ds, "height": hs, "ids": ids, "type": types})

    def __len__(self):
        return len(self.superitems)

    def __contains__(self, superitem):
        return hash(superitem) in self.hash_to_index

    def __getitem__(self, i):
        return self.superitems[i]

    def __str__(self):
        return f"SuperitemPool(superitems={self.superitems})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def gen_superitems(
        cls,
        order,
        pallet_dims,
        max_vstacked=2,
        only_single=False,
        horizontal=True,
        horizontal_type="two-width",
    ):
        """
        Generate horizontal and vertical superitems and
        filter the ones exceeding the pallet dimensions
        """
        singles_removed = []
        items = Item.from_dataframe(order)
        superitems = cls._gen_single_items_superitems(items)
        if only_single:
            logger.info("Generating superitems with only single items")
            return superitems, singles_removed
        if horizontal:
            logger.info(f"Generating horizontal superitems of type '{horizontal_type}'")
            superitems += cls._gen_superitems_horizontal(superitems, htype=horizontal_type)
            superitems, singles_removed = cls._drop_singles_in_horizontal(superitems)
        logger.info(f"Generating vertical superitems with maximum stacking of {max_vstacked}")
        superitems += cls._gen_superitems_vertical(superitems, max_vstacked)
        logger.info(f"Generated {len(superitems)} superitems")
        superitems = cls._filter_superitems(superitems, pallet_dims)
        logger.info(f"Remaining superitems after filtering by pallet dimensions: {len(superitems)}")
        return superitems, singles_removed

    @classmethod
    def _gen_single_items_superitems(cls, items):
        """
        Generate superitems with a single item
        """
        superitems = [SingleItemSuperitem([i]) for i in items]
        logger.debug(f"Generated {len(superitems)} superitems with a single item")
        return superitems

    @classmethod
    def _gen_superitems_horizontal(cls, items, htype="two-width"):
        """
        Horizontally stack groups of 2 and 4 items with the same
        dimensions to form single superitems
        """
        assert htype in (
            "all",
            "two-width",
            "two-depth",
            "four",
        ), "Unsupported horizontal superitem type"

        # Get items having the exact same dimensions
        dims = [(i.width, i.depth, i.height) for i in items]
        indexes = list(range(len(dims)))
        same_dims = defaultdict(list)
        for k, v in zip(dims, indexes):
            same_dims[k].append(v)

        # Extract candidate groups made up of 2 and 4 items
        two_slices, four_slices = [], []
        for _, indexes in same_dims.items():
            two_slices += [
                (items[indexes[i]], items[indexes[i + 1]]) for i in range(0, len(indexes) - 1, 2)
            ]
            four_slices += [
                (
                    items[indexes[i]],
                    items[indexes[i + 1]],
                    items[indexes[i + 2]],
                    items[indexes[i + 3]],
                )
                for i in range(0, len(indexes) - 3, 4)
            ]

        # Generate 2-items horizontal superitems
        two_superitems = []
        for slice in two_slices:
            if htype in ("all", "two-width"):
                two_superitems += [TwoHorizontalSuperitemWidth(slice)]
            elif htype in ("all", "two-depth"):
                two_superitems += [TwoHorizontalSuperitemDepth(slice)]
        logger.debug(f"Generated {len(two_superitems)} horizontal superitems with 2 items")

        # Generate 4-items horizontal superitems
        four_superitems = []
        for slice in four_slices:
            if htype in ("all", "four"):
                four_superitems += [FourHorizontalSuperitem(slice)]
        logger.debug(f"Generated {len(four_superitems)} horizontal superitems with 4 items")

        return two_superitems + four_superitems

    @classmethod
    def _drop_singles_in_horizontal(cls, superitems):
        """
        Remove single item superitems that appear in at least
        one horizontal superitem
        """
        # For each horizontal superitem, collect its components
        to_remove, removed = [], []
        for s in superitems:
            if isinstance(s, HorizontalSuperitem):
                ids = s.id
                for i, o in enumerate(superitems):
                    if isinstance(o, SingleItemSuperitem) and o.id[0] in ids:
                        to_remove += [i]
                        removed += [o]

        # Remove single item superitems in reverse order
        # to avoid indexing issues
        for i in sorted(to_remove, reverse=True):
            superitems.pop(i)

        return superitems, removed

    @classmethod
    def _gen_superitems_vertical(cls, superitems, max_vstacked, tol=0.7):
        """
        Divide superitems by width-depth ratio and vertically stack each group
        """
        assert tol >= 0.0, "Tolerance must be non-negative"
        assert max_vstacked > 1, "Maximum number of stacked items must be greater than 1"

        def _gen_superitems_vertical_subgroup(superitems):
            """
            Vertically stack groups of >= 2 items or superitems with the
            same dimensions to form a taller superitem
            """
            # Add the "width * depth" column and sort superitems
            # in ascending order by that dimension
            wd = [s.width * s.depth for s in superitems]
            superitems = [superitems[i] for i in argsort(wd)]

            # Extract candidate groups made up of >= 2 items or superitems
            slices = []
            for s in range(2, max_vstacked + 1):
                for i in range(0, len(superitems) - (s - 1), s):
                    good = True
                    for j in range(1, s, 1):
                        if (
                            superitems[i + j].width * superitems[i + j].depth
                            >= superitems[i].width * superitems[i].depth
                        ) and (
                            superitems[i].width * superitems[i].depth
                            <= tol * superitems[i + j].width * superitems[i + j].depth
                        ):
                            good = False
                            break
                    if good:
                        slices += [tuple(superitems[i + j] for j in range(s))]

            # Generate vertical superitems
            subgroup_vertical = []
            for slice in slices:
                subgroup_vertical += [VerticalSuperitem(slice)]

            return subgroup_vertical

        # Generate vertical superitems based on their aspect ratio
        wide, deep = [], []
        for s in superitems:
            if s.width / s.depth >= 1:
                wide.append(s)
            else:
                deep.append(s)
        wide_superitems = _gen_superitems_vertical_subgroup(wide)
        logger.debug(f"Generated {len(wide_superitems)} wide vertical superitems")
        deep_superitems = _gen_superitems_vertical_subgroup(deep)
        logger.debug(f"Generated {len(deep_superitems)} deep vertical superitems")
        return wide_superitems + deep_superitems

    @classmethod
    def _filter_superitems(cls, superitems, pallet_dims):
        """
        Keep only those superitems that do not exceed the
        pallet capacity
        """
        return [
            s
            for s in superitems
            if s.width <= pallet_dims.width
            and s.depth <= pallet_dims.depth
            and s.height <= pallet_dims.height
        ]


class Layer:
    """
    A layer represents the placement of a collection of
    items or superitems having similar heights
    """

    def __init__(self, superitems_pool, superitems_coords, pallet_dims):
        self.superitems_pool = superitems_pool
        self.superitems_coords = superitems_coords
        self.pallet_dims = pallet_dims

    @property
    def height(self):
        """
        Return the height of the current layer
        """
        return self.superitems_pool.get_max_height()

    @property
    def volume(self):
        """
        Return the sum of the items volumes in the layer
        """
        return sum(s.volume for s in self.superitems_pool)

    @property
    def area(self):
        """
        Return the sum of the items areas in the layer
        """
        return sum(s.width * s.depth for s in self.superitems_pool)

    def is_empty(self):
        """
        Return True if the current layer has no superitems, otherwise False
        """
        return len(self.superitems_pool) == 0 and len(self.superitems_coords) == 0

    def subset(self, superitem_indices):
        """
        Return a new layer with only the given superitems
        """
        new_spool = self.superitems_pool.subset(superitem_indices)
        new_scoords = [c for i, c in enumerate(self.superitems_coords) if i in superitem_indices]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def difference(self, superitem_indices):
        """
        Return a new layer without the given superitems
        """
        new_spool = self.superitems_pool.difference(superitem_indices)
        new_scoords = [
            c for i, c in enumerate(self.superitems_coords) if i not in superitem_indices
        ]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def get_items_coords(self, z=0):
        """
        Return a dictionary having as key the item id and as value
        the item coordinates in the layer
        """
        items_coords = dict()
        for s, c in zip(self.superitems_pool, self.superitems_coords):
            coords = s.get_items_coords(width=c.x, depth=c.y, height=z)
            duplicates = duplicate_keys([items_coords, coords])
            if len(duplicates) > 0:
                logger.error(f"Item repetition in the same layer, Items id:{duplicates}")
            items_coords = {**items_coords, **coords}
        return items_coords

    def get_items_dims(self):
        """
        Return a dictionary having as key the item id and as value
        the item dimensions in the layer
        """
        items_dims = dict()
        for s in self.superitems_pool:
            dims = s.get_items_dims()
            duplicates = duplicate_keys([items_dims, dims])
            if len(duplicates) > 0:
                logger.error(f"Item repetition in the same layer, Items id:{duplicates}")
            items_dims = {**items_dims, **dims}
        return items_dims

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer
        """
        return self.superitems_pool.get_unique_item_ids()

    def get_density(self, two_dims=False):
        """
        Compute the 2D/3D density of the layer
        """
        return (
            self.volume / (self.pallet_dims.area * self.height)
            if not two_dims
            else self.area / self.pallet_dims.area
        )

    def remove(self, superitem):
        """
        Return a new layer without the given superitem
        """
        new_spool = SuperitemPool(
            superitems=[s for s in self.superitems_pool if s != superitem]
        )
        new_scoords = [
            c
            for i, c in enumerate(self.superitems_coords)
            if i != self.superitems_pool.get_index(superitem)
        ]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def get_superitems_containing_item(self, item_id):
        """
        Return a list of superitems containing the given raw item
        """
        return self.superitems_pool.get_superitems_containing_item(item_id)

    def rearrange(self):
        """
        Apply maxrects over superitems in layer
        """
        return maxrects_single_layer_offline(self.superitems_pool, self.pallet_dims)

    def plot(self, ax=None, height=0):
        """
        Plot items in the current layer in the given plot or  in a new 3D plot
        having a maximum height given by the height of the layer
        """
        if ax is None:
            ax = get_pallet_plot(
                Dimension(self.pallet_dims.width, self.pallet_dims.depth, self.height)
            )
        items_coords = self.get_items_coords(z=height)
        items_dims = self.get_items_dims()
        for item_id in items_coords.keys():
            coords = items_coords[item_id]
            dims = items_dims[item_id]
            ax = plot_product(ax, item_id, coords, dims)
        return ax

    def to_dataframe(self, z=0):
        """
        Convert the current layer to a Pandas DataFrame
        """
        items_coords = self.get_items_coords()
        items_dims = self.get_items_dims()
        keys = list(items_coords.keys())
        xs = [items_coords[k].x for k in keys]
        ys = [items_coords[k].y for k in keys]
        zs = [items_coords[k].z + z for k in keys]
        ws = [items_dims[k].width for k in keys]
        ds = [items_dims[k].depth for k in keys]
        hs = [items_dims[k].height for k in keys]
        return pd.DataFrame(
            {
                "item": keys,
                "x": xs,
                "y": ys,
                "z": zs,
                "width": ws,
                "depth": ds,
                "height": hs,
            }
        )

    def __str__(self):
        return f"Layer(height={self.height}, ids={self.superitems_pool.get_unique_item_ids()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_items_coords() == other.get_items_coords()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.superitems_pool)

    def __contains__(self, superitem):
        return superitem in self.superitems_pool

    def __hash__(self):
        s_hashes = [hash(s) for s in self.superitems_pool]
        c_hashes = [hash(c) for c in self.superitems_coords]
        strs = [f"{s_hashes[i]}/{c_hashes[i]}" for i in argsort(s_hashes)]
        return hash("-".join(strs))


class LayerPool:
    """
    A layer pool is a collection of layers
    """

    def __init__(self, superitems_pool, pallet_dims, layers=None, add_single=False):
        self.superitems_pool = superitems_pool
        self.pallet_dims = pallet_dims
        self.layers = layers or []
        self.hash_to_index = self._get_hash_to_index()

        if add_single:
            self._add_single_layers()

    def _get_hash_to_index(self):
        """
        Compute a mapping for all layers in the pool with key the
        hash of the layer and value its index in the pool
        """
        return {hash(l): i for i, l in enumerate(self.layers)}

    def _add_single_layers(self):
        """
        Add one layer for each superitem that only
        contains that superitem
        """
        for superitem in self.superitems_pool:
            self.add(
                Layer(
                    SuperitemPool([superitem]),
                    [Coordinate(x=0, y=0)],
                    self.pallet_dims,
                )
            )

    def subset(self, layer_indices):
        """
        Return a new layer pool with the given subset of layers
        and the same superitems pool
        """
        layers = [l for i, l in enumerate(self.layers) if i in layer_indices]
        return LayerPool(self.superitems_pool, self.pallet_dims, layers=layers)

    def difference(self, layer_indices):
        """
        Return a new layer pool without the given subset of layers
        and the same superitems pool
        """
        layers = [l for i, l in enumerate(self.layers) if i not in layer_indices]
        return LayerPool(self.superitems_pool, self.pallet_dims, layers=layers)

    def get_ol(self):
        """
        Return a numpy array ol s.t. ol[l] = h iff
        layer l has height h
        """
        return np.array([layer.height for layer in self.layers], dtype=int)

    def get_zsl(self):
        """
        Return a binary matrix zsl s.t. zsl[s, l] = 1 iff
        superitem s is in layer l
        """
        zsl = np.zeros((len(self.superitems_pool), len(self.layers)), dtype=int)
        for s, superitem in enumerate(self.superitems_pool):
            for l, layer in enumerate(self.layers):
                if superitem in layer:
                    zsl[s, l] = 1
        return zsl

    def add(self, layer):
        """
        Add the given layer to the current pool
        """
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        l_hash = hash(layer)
        if l_hash not in self.hash_to_index:
            self.layers.append(layer)
            self.hash_to_index[l_hash] = len(self.layers) - 1

    def extend(self, layer_pool):
        """
        Extend the current pool with the given one
        """
        assert isinstance(
            layer_pool, LayerPool
        ), "The given set of layers should be an instance of the LayerPool class"
        check_dims = layer_pool.pallet_dims == self.pallet_dims
        assert check_dims, "The given LayerPool is defined over different pallet dimensions"
        for layer in layer_pool:
            self.add(layer)
        self.superitems_pool.extend(layer_pool.superitems_pool)

    def remove(self, layer):
        """
        Remove the given Layer from the LayerPool
        """
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        l_hash = hash(layer)
        if l_hash in self.hash_to_index:
            del self.layers[self.hash_to_index[l_hash]]
            self.hash_to_index = self._get_hash_to_index()

    def replace(self, i, layer):
        """
        Replace layer at index i with the given layer
        """
        assert i in range(len(self.layers)), "Index out of bounds"
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        del self.hash_to_index[hash(self.layers[i])]
        self.hash_to_index[hash(layer)] = i
        self.layers[i] = layer

    def pop(self, i):
        """
        Remove the layer at the given index from the pool
        """
        self.remove(self.layers[i])

    def get_unique_items_ids(self):
        """
        Return the flattened list of item ids inside the layer pool
        """
        return self.superitems_pool.get_unique_item_ids()

    def get_densities(self, two_dims=False):
        """
        Compute the 2D/3D density of each layer in the pool
        """
        return [layer.get_density(two_dims=two_dims) for layer in self.layers]

    def sort_by_densities(self, two_dims=False):
        """
        Sort layers in the pool by decreasing density
        """
        densities = self.get_densities(two_dims=two_dims)
        sorted_indices = argsort(densities, reverse=True)
        self.layers = [self.layers[i] for i in sorted_indices]

    def discard_by_densities(self, min_density=0.5, two_dims=False):
        """
        Sort layers by densities and keep only those with a
        density greater than or equal to the given minimum
        """
        assert min_density >= 0.0, "Density tolerance must be non-negative"
        self.sort_by_densities(two_dims=two_dims)
        densities = self.get_densities(two_dims=two_dims)
        last_index = -1
        for i, d in enumerate(densities):
            if d >= min_density:
                last_index = i
            else:
                break
        return self.subset(list(range(last_index + 1)))

    def discard_by_coverage(self, max_coverage_all=3, max_coverage_single=3):
        """
        Post-process layers by their item coverage
        """
        assert max_coverage_all > 0, "Maximum number of covered items in all layers must be > 0"
        assert (
            max_coverage_single > 0
        ), "Maximum number of covered items in a single layer must be > 0"
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [0] * len(all_item_ids)))
        layers_to_select = []
        for l, layer in enumerate(self.layers):
            to_select = True
            already_covered = 0

            # Stop when all items are covered
            if all([c > 0 for c in item_coverage.values()]):
                break

            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                # If at least one item in the layer was already selected
                # more times than the maximum allowed value, then such layer
                # is to be discarded
                if item_coverage[item] >= max_coverage_all:
                    to_select = False
                    break

                # If at least `max_coverage_single` items in the layer are already covered
                # by previously selected layers, then such layer is to be discarded
                if item_coverage[item] > 0:
                    already_covered += 1
                if already_covered >= max_coverage_single:
                    to_select = False
                    break

            # If the layer is selected, increase item coverage
            # for each item in such layer and add it to the pool
            # of selected layers
            if to_select:
                layers_to_select += [l]
                for item in item_ids:
                    item_coverage[item] += 1

        return self.subset(layers_to_select)

    def remove_duplicated_items(self, min_density=0.5, two_dims=False):
        """
        Keep items that are covered multiple times only
        in the layers with the highest densities
        """
        assert min_density >= 0.0, "Density tolerance must be non-negative"
        selected_layers = copy.deepcopy(self)
        all_item_ids = selected_layers.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        edited_layers, to_remove = set(), set()
        for l in range(len(selected_layers)):
            layer = selected_layers[l]
            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                duplicated_superitems, duplicated_indices = layer.get_superitems_containing_item(
                    item
                )
                # Remove superitems in different layers containing the same item
                # (remove the ones in less dense layers)
                if item_coverage[item]:
                    edited_layers.add(l)
                    layer = layer.difference(duplicated_indices)
                # Remove superitems in the same layer containing the same item
                # (remove the ones with less volume)
                elif len(duplicated_indices) > 1:
                    edited_layers.add(l)
                    duplicated_volumes = [s.volume for s in duplicated_superitems]
                    layer = layer.difference(
                        [duplicated_indices[i] for i in argsort(duplicated_volumes)[:-1]]
                    )

            if l in edited_layers:
                # Flag the layer if it doesn't respect the minimum density
                density = layer.get_density(two_dims=two_dims)
                if density < min_density or density == 0:
                    to_remove.add(l)
                # Replace the original layer with the edited one
                else:
                    selected_layers.replace(l, layer)

            # Update item coverage
            if l not in to_remove:
                item_ids = selected_layers[l].get_unique_items_ids()
                for item in item_ids:
                    item_coverage[item] = True

        # Rearrange layers in which at least one superitem was removed
        for l in edited_layers:
            if l not in to_remove:
                layer = selected_layers[l].rearrange()
                if layer is not None:
                    selected_layers[l] = layer
                else:
                    logger.error(f"After removing duplicated items couldn't rearrange layer {l}")

        # Removing layers last to first to avoid indexing errors
        for l in sorted(to_remove, reverse=True):
            selected_layers.pop(l)

        return selected_layers

    def remove_empty_layers(self):
        """
        Check and remove layers without any items
        """
        not_empty_layers = []
        for l, layer in enumerate(self.layers):
            if not layer.is_empty():
                not_empty_layers.append(l)
        return self.subset(not_empty_layers)

    def filter_layers(
        self, min_density=0.5, two_dims=False, max_coverage_all=3, max_coverage_single=3
    ):
        """
        Perform post-processing steps to select the best layers in the pool
        """
        logger.info(f"Filtering {len(self)} generated layers")
        new_pool = self.discard_by_densities(min_density=min_density, two_dims=two_dims)
        logger.debug(f"Remaining {len(new_pool)} layers after discarding by {min_density} density")
        new_pool = new_pool.discard_by_coverage(
            max_coverage_all=max_coverage_all, max_coverage_single=max_coverage_single
        )
        logger.debug(
            f"Remaining {len(new_pool)} layers after discarding by coverage "
            f"(all: {max_coverage_all}, single: {max_coverage_single})"
        )
        new_pool = new_pool.remove_duplicated_items(min_density=min_density, two_dims=two_dims)
        logger.debug(f"Remaining {len(new_pool)} layers after removing duplicated items")
        new_pool = new_pool.remove_empty_layers()
        logger.debug(f"Remaining {len(new_pool)} layers after removing the empty ones")
        new_pool.sort_by_densities(two_dims=two_dims)
        return new_pool

    def item_coverage(self):
        """
        Return a dictionary {i: T/F} identifying whether or not
        item i is included in a layer in the pool
        """
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        for layer in self.layers:
            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                item_coverage[item] = True
        return item_coverage

    def not_covered_single_superitems(self, singles_removed=None):
        """
        Return a list of single item superitems that are not present in the pool
        """
        # Get items not covered in the layer pool
        item_coverage = self.item_coverage()
        not_covered_ids = [k for k, v in item_coverage.items() if not v]
        not_covered = set()
        for s in self.superitems_pool:
            for i in not_covered_ids:
                if s.id == [i]:
                    not_covered.add(s)

        # Add not covered single items that were removed due to
        # layer filtering of horizontal superitems
        singles_removed = singles_removed or []
        for s in singles_removed:
            if s.id[0] not in item_coverage:
                not_covered.add(s)

        return list(not_covered)

    def not_covered_superitems(self):
        """
        Return a list of superitems which are not present in any layer
        """
        covered_spool = SuperitemPool(superitems=None)
        for l in self.layers:
            covered_spool.extend(l.superitems_pool)

        return [s for s in self.superitems_pool if covered_spool.get_index(s) is None]

    def get_heights(self):
        """
        Return the list of layer heights in the pool
        """
        return [l.height for l in self.layers]

    def get_areas(self):
        """
        Return the list of layer areas in the pool
        """
        return [l.area for l in self.layers]

    def get_volumes(self):
        """
        Return the list of layer volumes in the pool
        """
        return [l.volume for l in self.layers]

    def to_dataframe(self, zs=None):
        """
        Convert the layer pool to a Pandas DataFrame
        """
        if len(self) == 0:
            return pd.DataFrame()
        if zs is None:
            zs = [0] * len(self)
        dfs = []
        for i, layer in enumerate(self.layers):
            df = layer.to_dataframe(z=zs[i])
            df["layer"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0).reset_index(drop=True)

    def describe(self):
        """
        Return a DataFrame with stats about the current layer pool
        """
        ids = list(range(len(self.layers)))
        heights = self.get_heights()
        areas = self.get_areas()
        volumes = self.get_volumes()
        densities_2d = self.get_densities(two_dims=True)
        densities_3d = self.get_densities(two_dims=False)
        df = pd.DataFrame(
            zip(ids, heights, areas, volumes, densities_2d, densities_3d),
            columns=["layer", "height", "area", "volume", "2d_density", "3d_density"],
        )
        total = (
            df.agg(
                {
                    "height": np.sum,
                    "area": np.sum,
                    "volume": np.sum,
                    "2d_density": np.mean,
                    "3d_density": np.mean,
                }
            )
            .to_frame()
            .T
        )
        total["layer"] = "Total"
        return pd.concat((df, total), axis=0).reset_index(drop=True)

    def __str__(self):
        return f"LayerPool(layers={self.layers})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.layers)

    def __contains__(self, layer):
        return layer in self.layers

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, e):
        assert isinstance(e, Layer), "The given layer should be an instance of the Layer class"
        self.layers[i] = e

MAXRECTS_PACKING_STRATEGIES = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]

def maxrects_multiple_layers(superitems_pool, pallet_dims, add_single=True):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    return a layer pool with warm start placements
    """
    logger.debug("MR-ML-Offline starting")
    logger.debug(f"MR-ML-Offline {'used' if add_single else 'not_used'} as warm_start")
    logger.debug(f"MR-ML-Offline {len(superitems_pool)} superitems to place")

    # Return a layer with a single item if only one is present in the superitems pool
    if len(superitems_pool) == 1:
        layer_pool = LayerPool(superitems_pool, pallet_dims, add_single=True)
        uncovered = 0
    else:
        generated_pools = []
        for strategy in MAXRECTS_PACKING_STRATEGIES:
            # Build initial layer pool
            layer_pool = LayerPool(superitems_pool, pallet_dims, add_single=add_single)

            # Create the maxrects packing algorithm
            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.Global,
                pack_algo=strategy,
                sort_algo=SORT_AREA,
                rotation=False,
            )

            # Add an infinite number of layers (no upper bound)
            packer.add_bin(pallet_dims.width, pallet_dims.depth, count=float("inf"))

            # Add superitems to be packed
            ws, ds, _ = superitems_pool.get_superitems_dims()
            for i, (w, d) in enumerate(zip(ws, ds)):
                packer.add_rect(w, d, rid=i)

            # Start the packing procedure
            packer.pack()

            # Build a layer pool
            for layer in packer:
                spool, scoords = [], []
                for superitem in layer:
                    spool += [superitems_pool[superitem.rid]]
                    scoords += [Coordinate(superitem.x, superitem.y)]

                spool = SuperitemPool(superitems=spool)
                layer_pool.add(Layer(spool, scoords, pallet_dims))
                layer_pool.sort_by_densities(two_dims=False)

            # Add the layer pool to the list of generated pools
            generated_pools += [layer_pool]

        # Find the best layer pool by considering the number of placed superitems,
        # the number of generated layers and the density of each layer dense
        uncovered = [len(pool.not_covered_superitems()) for pool in generated_pools]
        n_layers = [len(pool) for pool in generated_pools]
        densities = [pool[0].get_density(two_dims=False) for pool in generated_pools]
        pool_indexes = argsort(list(zip(uncovered, n_layers, densities)), reverse=True)
        layer_pool = generated_pools[pool_indexes[0]]
        uncovered = uncovered[pool_indexes[0]]

    logger.debug(
        f"MR-ML-Offline generated {len(layer_pool)} layers with 3D densities {layer_pool.get_densities(two_dims=False)}"
    )
    logger.debug(
        f"MR-ML-Offline placed {len(superitems_pool) - uncovered}/{len(superitems_pool)} superitems"
    )
    return layer_pool


def maxrects_single_layer_offline(superitems_pool, pallet_dims, superitems_in_layer=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into,
    try to fit each superitem in a single layer (if not possible, return an error)
    """
    logger.debug("MR-SL-Offline starting")

    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    logger.debug(f"MR-SL-Offline {superitems_in_layer}/{len(superitems_pool)} superitems to place")

    # Iterate over each placement strategy
    ws, ds, _ = superitems_pool.get_superitems_dims()
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Offline,
            bin_algo=PackingBin.Global,
            pack_algo=strategy,
            sort_algo=SORT_AREA,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.depth, count=1)

        # Add superitems to be packed
        for i in superitems_in_layer:
            packer.add_rect(ws[i], ds[i], rid=i)

        # Start the packing procedure
        packer.pack()

        # Feasible packing with a single layer
        if len(packer) == 1 and len(packer[0]) == len(superitems_in_layer):
            spool = SuperitemPool(superitems=[superitems_pool[s.rid] for s in packer[0]])
            layer = Layer(
                spool, [Coordinate(s.x, s.y) for s in packer[0]], pallet_dims
            )
            logger.debug(
                f"MR-SL-Offline generated a new layer with {len(layer)} superitems "
                f"and {layer.get_density(two_dims=False)} 3D density"
            )
            return layer

    return None


def maxrects_single_layer_online(superitems_pool, pallet_dims, superitems_duals=None):
    """
    Given a superitems pool and the maximum dimensions to pack them into, try to fit
    the greatest number of superitems in a single layer following the given order
    """
    logger.debug("MR-SL-Online starting")

    # If no duals are given use superitems' heights as a fallback
    ws, ds, hs = superitems_pool.get_superitems_dims()
    if superitems_duals is None:
        superitems_duals = np.array(hs)

    # Sort rectangles by duals
    indexes = argsort(list(zip(superitems_duals, hs)), reverse=True)
    logger.debug(
        f"MR-SL-Online {sum(superitems_duals[i] > 0 for i in indexes)} non-zero duals to place"
    )

    # Iterate over each placement strategy
    generated_layers, num_duals = [], []
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Online,
            pack_algo=strategy,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.depth, count=1)

        # Online packing procedure
        n_packed, non_zero_packed, layer_height = 0, 0, 0
        for i in indexes:
            if superitems_duals[i] > 0 or hs[i] <= layer_height:
                packer.add_rect(ws[i], ds[i], i)
                if len(packer[0]) > n_packed:
                    n_packed = len(packer[0])
                    if superitems_duals[i] > 0:
                        non_zero_packed += 1
                    if hs[i] > layer_height:
                        layer_height = hs[i]
        num_duals += [non_zero_packed]

        # Build layer after packing
        spool, coords = [], []
        for s in packer[0]:
            spool += [superitems_pool[s.rid]]
            coords += [Coordinate(s.x, s.y)]
        layer = Layer(SuperitemPool(spool), coords, pallet_dims)
        generated_layers += [layer]

    # Find the best layer by taking into account the number of
    # placed superitems with non-zero duals and density
    layer_indexes = argsort(
        [
            (duals, layer.get_density(two_dims=False))
            for duals, layer in zip(num_duals, generated_layers)
        ],
        reverse=True,
    )
    layer = generated_layers[layer_indexes[0]]

    logger.debug(
        f"MR-SL-Online generated a new layer with {len(layer)} superitems "
        f"(of which {num_duals[layer_indexes[0]]} with non-zero dual) "
        f"and {layer.get_density(two_dims=False)} 3D density"
    )
    return layer
