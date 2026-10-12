"""Model orientation for the structural view.

The orientation is stored as three angles (degrees) about the X, Y and Z axes,
composed as ``R = Rz @ Ry @ Rx`` and applied about the centroid of the
reference (first analysed) frame.  The same matrix is used for every rendered
frame, so the orientation chosen on the preview holds for the whole movie.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_to_matrix(angles: Sequence[float]) -> np.ndarray:
    ax, ay, az = (float(v) for v in angles)
    return rz(az) @ ry(ay) @ rx(ax)


def wrap_degrees(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(a, -180.0) else a


def matrix_to_euler(r: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`euler_to_matrix` (degrees, each in (-180, 180])."""
    r = np.asarray(r, dtype=np.float64)
    sy = -r[2, 0]
    sy = min(1.0, max(-1.0, sy))
    beta = np.arcsin(sy)
    if abs(np.cos(beta)) > 1e-8:
        alpha = np.arctan2(r[2, 1], r[2, 2])
        gamma = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: fold Z into X
        gamma = 0.0
        alpha = np.arctan2(-r[1, 2], r[1, 1])
    return tuple(wrap_degrees(float(np.degrees(v))) for v in (alpha, beta, gamma))


def drag(angles: Sequence[float], dx: float, dy: float,
         degrees_per_pixel: float = 0.5) -> Tuple[float, float, float]:
    """New angles after a mouse drag of (dx, dy) pixels (screen-space trackball).

    Horizontal motion spins the model about the screen's vertical axis and
    vertical motion about the horizontal axis (screen y grows downwards).
    """
    r = euler_to_matrix(angles)
    r = ry(dx * degrees_per_pixel) @ rx(dy * degrees_per_pixel) @ r
    return matrix_to_euler(r)


def project(points: np.ndarray, angles: Sequence[float], center: np.ndarray) -> np.ndarray:
    """Rotate points about ``center``; returns view coordinates (x right, y up, z to viewer)."""
    r = euler_to_matrix(angles)
    return (np.asarray(points, dtype=np.float64) - center) @ r.T


def format_angles(angles: Sequence[float]) -> str:
    deg = ", ".join(f"{a:+.1f}°" for a in angles)
    rad = ", ".join(f"{np.radians(a):+.4f}" for a in angles)
    return f"X/Y/Z = {deg}  ({rad} rad)"
