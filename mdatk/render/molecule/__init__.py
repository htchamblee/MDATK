"""3D molecular representations for the structural view.

``topology`` finds the atoms each style needs (backbone control points, sugar
and base rings) from the cached ``atoms`` analysis, ``geometry`` turns
coordinates into meshes, sticks and spheres with plain numpy, and the
renderers draw those primitives.
"""
