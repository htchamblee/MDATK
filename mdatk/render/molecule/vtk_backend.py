"""Optional VTK renderer for the 3D representations (``pip install vtk``).

Used only when ``renderer: vtk`` is chosen; the built-in software renderer is the
default.  VTK renders offscreen into an RGBA image with a transparent background, so
the density shadow and the glow halos drawn by matplotlib show around the molecule.

* Only ``vtkmodules`` is used (no pyvista).  Spheres and sticks are GPU
  impostors; meshes are ordinary polydata.
* The view rotation is the actors' user matrix, the camera is orthographic and
  matched to the panel's axes limits, and the depth cue is applied in the
  fragment shader (blend toward the panel colour over the panel's depth range,
  as the software renderer does).
* Anti-aliasing: the silhouette alpha is smoothed like the software renderer's.
* OpenGL contexts belong to the thread that created them, so every thread
  (GUI main loop, video worker) gets its own scene; call :func:`release_thread`
  when a thread is done.
* Headless machines need EGL (e.g. ``libegl1 libegl-mesa0``) or OSMesa; without
  them VTK can crash inside ``Render()``.  :func:`available` therefore runs a
  small probe in a subprocess once (cached) before VTK is used in-process.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import Mesh, Spheres, Sticks
from .scene import Colored, Shading, depth_range

_PROBE: Dict[Tuple, Tuple[bool, str]] = {}
_SCENES: Dict[int, Tuple[threading.Thread, "Scene"]] = {}
_LOCK = threading.Lock()


def _probe_key() -> Tuple:
    return (os.environ.get("DISPLAY", ""), os.environ.get("WAYLAND_DISPLAY", ""),
            os.environ.get("VTK_DEFAULT_OPENGL_WINDOW", ""), sys.executable)


def available() -> Tuple[bool, str]:
    """(usable, reason).  The first call runs a 64x64 offscreen render in a subprocess."""
    if os.environ.get("MDATK_NO_VTK"):
        return False, "VTK disabled (MDATK_NO_VTK is set)"
    if importlib.util.find_spec("vtkmodules") is None:
        return False, "VTK is not installed (pip install vtk)"
    key = _probe_key()
    with _LOCK:
        if key in _PROBE:
            return _PROBE[key]
    try:
        out = subprocess.run([sys.executable, "-c",
                              "from mdatk.render.molecule.vtk_backend import _probe; _probe()"],
                             capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
        info = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() \
            else None
        if info and info.get("ok"):
            result = (True, f"VTK {info.get('version')} ({info.get('window')})")
        else:
            result = (False, "VTK cannot render here (no display, and no EGL/OSMesa libraries; on Linux "
                             "install libegl1 and libegl-mesa0)" + (f": exit code {out.returncode}"
                                                                     if out.returncode else ""))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result = (False, f"VTK probe failed: {exc}")
    with _LOCK:
        _PROBE[key] = result
    return result


def _probe() -> None:  # pragma: no cover - runs in a subprocess
    import vtkmodules
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow

    importlib.import_module("vtkmodules.vtkRenderingOpenGL2")  # registers the OpenGL classes

    rw = vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(64, 64)
    ren = vtkRenderer()
    ren.SetBackground(0.2, 0.4, 0.6)
    rw.AddRenderer(ren)
    rw.Render()
    arr = vtkUnsignedCharArray()
    rw.GetRGBACharPixelData(0, 0, 63, 63, 0, arr)
    ok = arr.GetNumberOfTuples() == 64 * 64
    print(json.dumps({"ok": bool(ok), "window": rw.GetClassName(),
                      "version": getattr(vtkmodules, "__version__", "?")}))


def release_thread() -> None:
    """Free the calling thread's VTK scene (call it from that thread)."""
    with _LOCK:
        entry = _SCENES.pop(threading.get_ident(), None)
    if entry is not None and entry[0] is threading.current_thread():
        entry[1].finalize()


class VtkRenderer:
    name = "vtk"
    draws_glow = False  # the panel keeps its glow halos and tints glowing residues

    def render(self, items: List[Colored], viewport, shading: Shading, quality: str = "high") -> np.ndarray:
        tid, me = threading.get_ident(), threading.current_thread()
        with _LOCK:
            entry = _SCENES.get(tid)
            if entry is None or entry[0] is not me:  # thread ids are reused: never share a context
                entry = _SCENES[tid] = (me, Scene())
        rgba = entry[1].render(items, viewport, shading)
        if quality != "fast":
            from scipy import ndimage

            a = rgba[..., 3].astype(np.float32)
            a3 = ndimage.uniform_filter(a, 3, mode="constant")
            rgba[..., 3] = np.where(np.abs(a3 - a) > 1, 0.5 * (a + a3), a).astype(np.uint8)
        return rgba


class Scene:
    """One offscreen window with reusable actors (shader compilation happens once per actor)."""

    def __init__(self):
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow

        importlib.import_module("vtkmodules.vtkRenderingOpenGL2")  # registers the OpenGL classes

        try:
            from vtkmodules.vtkCommonCore import vtkLogger

            vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
        except Exception:  # noqa: BLE001 - older VTK
            pass
        self.rw = vtkRenderWindow()
        self.rw.SetOffScreenRendering(1)
        self.rw.SetAlphaBitPlanes(1)
        self.rw.SetMultiSamples(0)
        self.ren = vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.SetBackgroundAlpha(0.0)
        self.ren.SetUseDepthPeeling(0)
        self.rw.AddRenderer(self.ren)
        self.view = vtkMatrix4x4()
        self.pools: Dict[str, list] = {"mesh": [], "spheres": [], "sticks": []}
        self.fog: Optional[Tuple[float, float, float, float]] = None
        self.size = (0, 0)

    def finalize(self) -> None:
        try:
            self.rw.Finalize()
        except Exception:  # noqa: BLE001
            pass

    # -- actors
    def _actor(self, kind: str, k: int):
        pool = self.pools[kind]
        while len(pool) <= k:
            pool.append(self._new_actor(kind))
        return pool[k]

    def _new_actor(self, kind: str):
        from vtkmodules.vtkCommonDataModel import vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLSphereMapper, vtkOpenGLStickMapper

        pd = vtkPolyData()
        if kind == "spheres":
            m = vtkOpenGLSphereMapper()
            m.SetScaleArray("radius")
        elif kind == "sticks":
            m = vtkOpenGLStickMapper()
            m.SetScaleArray("scale")
            m.SetOrientationArray("orient")
            m.SetSelectionIdArray("sel")
        else:
            m = vtkPolyDataMapper()
        m.SetInputData(pd)
        m.SetScalarModeToUsePointData()
        m.SetColorModeToDirectScalars()
        a = vtkActor()
        a.SetMapper(m)
        a.SetUserMatrix(self.view)
        p = a.GetProperty()
        p.SetAmbient(0.28)
        p.SetDiffuse(0.72)
        p.SetSpecular(0.22)
        p.SetSpecularPower(28)
        a.pd = pd
        a.fog = None
        self.ren.AddActor(a)
        return a

    def _set_fog(self, actor, fog) -> None:
        """Fog fraction ``clamp(a + b * depth01, 0, strength)`` toward ``rgb`` (uniforms, so
        only switching it on or off recompiles the shader)."""
        r, g, b, strength, fa, fb = fog
        sp = actor.GetShaderProperty()
        on = strength > 0
        if actor.fog != on:
            sp.ClearAllFragmentShaderReplacements()
            if on:
                sp.AddFragmentShaderReplacement(
                    "//VTK::Light::Impl", False,
                    "//VTK::Light::Impl\n  gl_FragData[0].rgb = mix(gl_FragData[0].rgb, fogColor, "
                    "clamp(fogA + fogB * gl_FragCoord.z, 0.0, fogMax));\n", False)
            actor.fog = on
        if on:
            u = sp.GetFragmentCustomUniforms()
            u.SetUniform3f("fogColor", [r, g, b])
            u.SetUniformf("fogA", fa)
            u.SetUniformf("fogB", fb)
            u.SetUniformf("fogMax", strength)

    # -- primitives
    def _points(self, pd, xyz, arrays, rgb):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray

        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(xyz, dtype=np.float32), deep=1))
        pd.SetPoints(pts)
        n = len(xyz)
        if pd.GetNumberOfVerts() != n:
            ca = vtkCellArray()
            ca.SetData(numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=1),
                       numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=1))
            pd.SetVerts(ca)
        pdata = pd.GetPointData()
        for name, a in arrays.items():
            arr = numpy_to_vtk(np.ascontiguousarray(a), deep=1)
            arr.SetName(name)
            pdata.AddArray(arr)
        pdata.SetScalars(numpy_to_vtk(_rgb8(rgb), deep=1))
        pd.Modified()

    def _mesh(self, pd, m: Mesh, rgb):
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray

        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(m.verts, dtype=np.float32), deep=1))
        pd.SetPoints(pts)
        n = len(m.tris)
        ca = vtkCellArray()
        ca.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * n + 1, 3, dtype=np.int64), deep=1),
                   numpy_to_vtkIdTypeArray(np.ascontiguousarray(m.tris, dtype=np.int64).ravel(), deep=1))
        pd.SetPolys(ca)
        pd.GetPointData().SetNormals(numpy_to_vtk(np.ascontiguousarray(m.normals, dtype=np.float32), deep=1))
        pd.GetPointData().SetScalars(numpy_to_vtk(_rgb8(rgb), deep=1))
        pd.Modified()

    # -- render
    def render(self, items: List[Colored], vp, shading: Shading) -> np.ndarray:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkCommonCore import vtkUnsignedCharArray

        w, h = vp.width, vp.height
        if self.size != (w, h):
            self.rw.SetSize(w, h)
            self.size = (w, h)
        for i in range(3):
            for j in range(3):
                self.view.SetElement(i, j, float(vp.R[i, j]))
        self.view.Modified()
        # orthographic camera matched to the axes (view coordinates: x right, y up, z to the
        # viewer), clipped to the drawn geometry
        zlo, zhi = depth_range(items, vp.R[2])
        near = 10.0
        eye = zhi + near
        far = near + (zhi - zlo)
        # fog over the panel's depth range: view z = eye - near - depth01 * (far - near)
        flo, fhi = vp.zrange
        span = max(fhi - flo, 1e-6)
        k = float(shading.fog_strength)
        fog = (*(float(c) for c in shading.fog_rgb[:3]), k,
               k * (fhi - eye + near) / span, k * (far - near) / span)
        used = {"mesh": 0, "spheres": 0, "sticks": 0}
        for it in items:
            p = it.prim
            if isinstance(p, Mesh):
                if not len(p.tris):
                    continue
                a = self._actor("mesh", used["mesh"])
                used["mesh"] += 1
                self._mesh(a.pd, p, it.rgb)
            elif isinstance(p, Spheres):
                if not len(p.xyz):
                    continue
                a = self._actor("spheres", used["spheres"])
                used["spheres"] += 1
                self._points(a.pd, p.xyz, {"radius": np.asarray(p.radius, np.float32)}, it.rgb)
            elif isinstance(p, Sticks):
                if not len(p.a):
                    continue
                a = self._actor("sticks", used["sticks"])
                used["sticks"] += 1
                v = p.b - p.a
                length = np.linalg.norm(v, axis=1)
                scale = np.stack([length, p.radius, p.radius], 1).astype(np.float32)
                self._points(a.pd, 0.5 * (p.a + p.b),
                             {"scale": scale, "orient": (v / (length[:, None] + 1e-9)).astype(np.float32),
                              "sel": np.zeros(len(length), np.uint32)}, it.rgb)
            else:
                continue
            a.GetProperty().SetOpacity(float(it.opacity))
            self._set_fog(a, fog)
            a.VisibilityOn()
        for kind, pool in self.pools.items():
            for a in pool[used[kind]:]:
                a.VisibilityOff()
        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        xmid, ymid = 0.5 * sum(vp.xlim), 0.5 * sum(vp.ylim)
        cam.SetFocalPoint(xmid, ymid, zlo)
        cam.SetPosition(xmid, ymid, eye)
        cam.SetViewUp(0.0, 1.0, 0.0)
        cam.SetParallelScale(0.5 * (vp.ylim[1] - vp.ylim[0]))
        cam.SetClippingRange(near, far)
        self.rw.Render()
        arr = vtkUnsignedCharArray()
        self.rw.GetRGBACharPixelData(0, 0, w - 1, h - 1, 0, arr)
        return np.ascontiguousarray(vtk_to_numpy(arr).reshape(h, w, 4)[::-1])  # VTK rows start at the bottom


def _rgb8(rgb) -> np.ndarray:
    return np.ascontiguousarray(np.clip(np.asarray(rgb, dtype=np.float64) * 255.0 + 0.5, 0, 255).astype(np.uint8))
