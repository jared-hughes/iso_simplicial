# Iso simplicial

**Note: development has moved to https://github.com/jared-hughes/manim/tree/implicit-plotting. This repository will not be updated.**

Given a function and its gradient, this constructs an isosurface based on the method in
[Manson, Josiah, and Scott Schaefer. "Isosurfaces over simplicial partitions of multiresolution grids." Computer Graphics Forum. Vol. 29. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2010.](https://people.engr.tamu.edu/schaefer/research/iso_simplicial.pdf)

The paper uses the terminology of a 2D isosurface embedded in 3D space, but this repository implements a 1D isosurface (a curve) embedded in 2D space. We are finding the level curve `f(x,y)=0`. When the paper talks about octrees, etc., we use quadtrees.
