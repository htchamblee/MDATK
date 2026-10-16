"""Built-in renderer: a vectorised numpy z-buffer rasteriser (no OpenGL, works headless).

Primitives are rasterised exactly in an orthographic view matched to the panel's axes:

* triangles by scanline spans with plane-linear depth and interpolated normals,
* spheres as per-pixel impostors (one sprite per radius and sub-pixel phase),
* sticks by analytic ray / cylinder intersection.

Every fragment carries a 64-bit key (quantised depth | pass | primitive), so one
``np.maximum.at`` keeps the front-most fragment per pixel; only those pixels are
shaded (ambient + Lambert + Phong highlight from a camera light).  A compositor then
applies the depth cue (fog toward the panel colour, as for the 2D trace), the glow
of residues near the target (a tint plus a soft halo), ink outlines at depth jumps
and alpha anti-aliasing of silhouettes, and returns a straight-alpha RGBA image
(row 0 at the top).  Representations with opacity < 1 are
drawn in a second pass in front of the opaque surface and blended over it.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np
from scipy import ndimage

from .geometry import Mesh, Spheres, Sticks
from .scene import Colored, Shading, depth_range

LIGHT = np.array([-0.35, 0.55, 0.76])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE, SPECULAR = 0.36, 0.64, 0.22
ZBITS = 31  # quantised depth bits
IDBITS = 27  # fragment index bits per pass
MAX_PASSES = 32  # 5 pass bits: ZBITS + 5 + IDBITS = 63
CULL_MIN = 64  # spheres per list from which hidden ones are culled
INK = np.array([0.12, 0.12, 0.12], np.float32)
# glow: halo radius / disc (Å), halo opacity, tint toward the glow colour
GLOW_RADIUS, GLOW_DISC, GLOW_ALPHA, GLOW_TINT = 2.0, 2.5, 0.75, 0.25


def shade(nrm: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Ambient + Lambert + Phong highlight (shininess 24) from a light near the camera."""
    nx, ny, nz = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    inv = 1 / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), np.float32(1e-6))
    lx, ly, lz = (np.float32(v) for v in LIGHT)
    ndl_raw = (nx * lx + ny * ly + nz * lz) * inv
    ndl = np.clip(ndl_raw, 0, 1)
    rv = np.clip(2 * ndl_raw * nz * inv - lz, 0, 1) * (ndl_raw > 0)  # reflect(-L, n) . view
    x2 = rv * rv
    x8 = (x2 * x2) ** 2
    out = rgb * (np.float32(AMBIENT) + np.float32(DIFFUSE) * ndl)[:, None]
    out += (np.float32(SPECULAR) * x8 * x8 * x8)[:, None]
    return out


def _spans(G, y0, y1, W):
    """Fragments (prim, x, y) of convex regions given as half-planes a*x + b*y + c >= 0
    (G: (P, K, 3)), rows y0..y1 per primitive, tested at pixel centres."""
    nrow = (y1 - y0 + 1).astype(np.int64)
    prim = np.repeat(np.arange(len(G), dtype=np.int64), nrow)
    start = np.cumsum(nrow) - nrow
    y = y0[prim] + (np.arange(int(nrow.sum()), dtype=np.int64) - np.repeat(start, nrow))
    yc = y.astype(G.dtype) + 0.5
    big = G.dtype.type(1e9)
    lo = np.full(len(y), -big, G.dtype)
    hi = np.full(len(y), big, G.dtype)
    for k in range(G.shape[1]):
        a = np.ascontiguousarray(G[:, k, 0])
        pos, neg = a > 1e-9, a < -1e-9
        ia = np.where(pos | neg, 1 / np.where(pos | neg, a, 1), 0).astype(G.dtype)
        b = (G[:, k, 1] * ia)[prim]
        c = (G[:, k, 2] * ia)[prim]
        bound = -(b * yc + c)
        pk, nk = pos[prim], neg[prim]
        np.maximum(lo, np.where(pk, bound, -big), out=lo)
        np.minimum(hi, np.where(nk, bound, big), out=hi)
        flat = ~(pk | nk)
        if flat.any():  # horizontal constraint: the whole row is in or out
            e = G[:, k, 1][prim[flat]] * yc[flat] + G[:, k, 2][prim[flat]]
            hi[np.flatnonzero(flat)[e < 0]] = -big
    x0 = np.maximum(np.ceil(lo - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.floor(hi - 0.5), W - 1).astype(np.int64)
    cnt = np.maximum(x1 - x0 + 1, 0)
    rows = np.flatnonzero(cnt)
    cnt = cnt[rows]
    fprim = np.repeat(prim[rows], cnt)
    fy = np.repeat(y[rows], cnt)
    st = np.cumsum(cnt) - cnt
    fx = np.repeat(x0[rows] - st, cnt) + np.arange(int(cnt.sum()), dtype=np.int64)
    return fprim, fx, fy


class _Cam:
    """Pixel mapping of the panel's axes (``scale`` < 1 renders a smaller image) and the
    depth range of the drawn geometry (keys) and of the panel (fog)."""

    def __init__(self, vp, items: List[Colored], scale: float = 1.0):
        self.R = vp.R
        self.xlim, self.ylim = vp.xlim, vp.ylim
        self.W, self.H = max(2, int(round(vp.width * scale))), max(2, int(round(vp.height * scale)))
        self.s = self.W / (self.xlim[1] - self.xlim[0])  # pixels per Å
        self.sy = self.H / (self.ylim[1] - self.ylim[0])
        self.zmin, self.zmax = depth_range(items, self.R[2])
        self.fog_lo, self.fog_hi = vp.zrange

    def screen(self, xyz):
        v = np.asarray(xyz, np.float64) @ self.R.T
        px = (v[:, 0] - self.xlim[0]) * self.s
        py = (self.ylim[1] - v[:, 1]) * self.sy
        return px.astype(np.float32), py.astype(np.float32), v[:, 2].astype(np.float32)


_LOCAL = threading.local()


def _buffer(n: int) -> np.ndarray:
    """``n`` z-buffer keys set to -1.  The memory is reused per thread (fresh pages are
    slow); a pass is resolved before the next one starts, so one buffer suffices."""
    buf = getattr(_LOCAL, "zbuf", None)
    if buf is None or len(buf) < n:
        buf = _LOCAL.zbuf = np.empty(n, np.int64)
    out = buf[:n]
    out.fill(-1)
    return out


def _tiles(cam, X, Y, rp, T):
    """Per sphere: the first tile column / row its square footprint touches, the last ones,
    and the largest footprint (in tiles)."""
    tx0, tx1 = np.floor((X - rp) / T).astype(np.int64), np.floor((X + rp) / T).astype(np.int64)
    ty0, ty1 = np.floor((Y - rp) / T).astype(np.int64), np.floor((Y + rp) / T).astype(np.int64)
    return tx0, tx1, ty0, ty1, int(max((tx1 - tx0).max(), (ty1 - ty0).max())) + 1


def _unoccluded(cam, X, Y, Z, rad) -> np.ndarray:
    """Spheres that may be visible (conservative tile occlusion culling).

    For every screen tile, the depth each of its pixels is guaranteed to reach is the
    largest depth any sphere has over the whole tile (spheres covering the tile, shrunk by
    half a pixel for the sprites' sub-pixel placement).  A sphere whose front point lies
    behind that depth on every tile it touches cannot show and is skipped; the image is
    identical with and without culling."""
    n = len(X)
    if n < CULL_MIN:
        return np.ones(n, bool)
    X, Y, Z = X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)
    rp = rad.astype(np.float64) * cam.s
    T = max(8, int(np.ceil(rp.max() / 4)))  # at most ~10 tiles across a sphere
    tw, th = -(-cam.W // T), -(-cam.H // T)
    tx0, tx1, ty0, ty1, span = _tiles(cam, X, Y, rp, T)
    occ = np.full(tw * th, -np.inf)
    ro = np.maximum(rp - 0.5, 0)
    for dy in range(span):
        ty = ty0 + dy
        ey = np.maximum(np.abs(ty * T - Y), np.abs((ty + 1) * T - Y))  # farthest tile edge
        for dx in range(span):
            tx = tx0 + dx
            ex = np.maximum(np.abs(tx * T - X), np.abs((tx + 1) * T - X))
            r2 = ex * ex + ey * ey
            full = (tx <= tx1) & (ty <= ty1) & (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th) & (r2 < ro * ro)
            if full.any():
                np.maximum.at(occ, (ty * tw + tx)[full], (Z + np.sqrt(np.maximum(ro * ro - r2, 0)) / cam.s)[full])
    front = Z + rad
    hidden = np.ones(n, bool)
    for dy in range(span):
        ty = ty0 + dy
        for dx in range(span):
            tx = tx0 + dx
            on = (tx <= tx1) & (ty <= ty1) & (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th)
            t = np.where(on, ty * tw + tx, 0)
            hidden &= ~on | (occ[t] > front)  # tiles off the image draw nothing
    return ~hidden


class _ZPass:
    """One z-buffer and the primitive passes written into it."""

    def __init__(self, cam: _Cam):
        self.cam = cam
        self.zbuf = _buffer(cam.W * cam.H)
        self.passes: list = []

    def _submit(self, pix, z, attrs):
        cam = self.cam
        zq = (np.clip((z - cam.zmin) / (cam.zmax - cam.zmin), 0, 1) * ((1 << ZBITS) - 1)).astype(np.int64)
        step = (1 << IDBITS) - 1
        for offset in range(0, len(pix), step):  # one pass per 2**27 fragments
            pid = len(self.passes)
            if pid >= MAX_PASSES:
                raise RuntimeError("scene too large for the built-in renderer")
            n = min(step, len(pix) - offset)
            key = (zq[offset:offset + n] << (IDBITS + 5)) | (np.int64(pid) << IDBITS) | np.arange(n, dtype=np.int64)
            np.maximum.at(self.zbuf, pix[offset:offset + n], key)
            self.passes.append((attrs, offset))

    # -- primitives
    def mesh(self, m: Mesh, rgb):
        cam = self.cam
        X, Y, Z = cam.screen(m.verts)
        Nv = (np.asarray(m.normals, np.float64) @ cam.R.T).astype(np.float32)
        t = np.asarray(m.tris, np.int64)
        t = t[(Nv[t, 2] > -0.05).any(1)]  # back faces (all vertex normals facing away)
        if not len(t):
            return
        x0, x1, x2 = (X[t[:, k]].astype(np.float64) for k in range(3))
        y0, y1, y2 = (Y[t[:, k]].astype(np.float64) for k in range(3))
        A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        bx0 = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5), 0, cam.W - 1)
        bx1 = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5), 0, cam.W - 1)
        by0 = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5), 0, cam.H - 1)
        by1 = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5), 0, cam.H - 1)
        sel = np.flatnonzero((np.abs(A) > 1e-4) & (bx1 >= bx0) & (by1 >= by0))
        if not len(sel):
            return
        t, A = t[sel], A[sel]
        x0, x1, x2, y0, y1, y2 = x0[sel], x1[sel], x2[sel], y0[sel], y1[sel], y2[sel]
        inv = 1.0 / A
        a0, b0, c0 = (y1 - y2) * inv, (x2 - x1) * inv, (x1 * y2 - x2 * y1) * inv
        a1, b1, c1 = (y2 - y0) * inv, (x0 - x2) * inv, (x2 * y0 - x0 * y2) * inv
        G = np.empty((len(t), 3, 3), np.float64)
        G[:, 0] = np.stack([a0, b0, c0], 1)
        G[:, 1] = np.stack([a1, b1, c1], 1)
        G[:, 2] = np.stack([-a0 - a1, -b0 - b1, 1 - c0 - c1], 1)
        G[:, :, 2] += 0.02 * np.hypot(G[:, :, 0], G[:, :, 1])  # slightly conservative: no cracks
        prim, px, py = _spans(G, by0[sel].astype(np.int64), by1[sel].astype(np.int64), cam.W)
        fx, fy = px + 0.5, py + 0.5
        l0 = (a0[prim] * fx + b0[prim] * fy + c0[prim]).astype(np.float32)
        l1 = (a1[prim] * fx + b1[prim] * fy + c1[prim]).astype(np.float32)
        z0, z1, z2 = (Z[t[:, k]].astype(np.float64) for k in range(3))  # depth is plane-linear too
        za, zb = a0 * (z0 - z2) + a1 * (z1 - z2), b0 * (z0 - z2) + b1 * (z1 - z2)
        zc = z2 + c0 * (z0 - z2) + c1 * (z1 - z2)
        z = (za[prim] * fx + zb[prim] * fy + zc[prim]).astype(np.float32)
        cols = tuple(np.ascontiguousarray(t[:, k]) for k in range(3))
        nrm = tuple(np.ascontiguousarray(Nv[:, k]) for k in range(3))
        self._submit(py * cam.W + px, z, ("mesh", nrm, cols, prim, l0, l1, z, np.asarray(rgb, np.float32),
                                          np.asarray(m.owner, np.int64)))

    def spheres(self, s: Spheres, rgb):  # noqa: C901 - one sprite table per radius / phase
        cam = self.cam
        X, Y, Z = cam.screen(s.xyz)
        rad = np.asarray(s.radius, np.float32)
        keep = _unoccluded(cam, X, Y, Z, rad)
        cx = np.floor(X).astype(np.int64)
        cy = np.floor(Y).astype(np.int64)
        q = np.minimum(((X - cx) * 2).astype(np.int64), 1) * 2 + np.minimum(((Y - cy) * 2).astype(np.int64), 1)
        pix_l, z_l, prim_l, spr_l, tab = [], [], [], [], []
        n_tab = 0
        for rv in np.unique(rad):
            rp = float(rv) * cam.s
            n = int(np.ceil(rp)) + 1
            oy, ox = np.mgrid[-n:n + 1, -n:n + 1]
            for qq in range(4):
                sel = np.flatnonzero((rad == rv) & (q == qq) & keep)
                if not len(sel):
                    continue
                sx, sy = (0.25 if qq < 2 else 0.75), (0.25 if qq % 2 == 0 else 0.75)
                dx, dy = ox + 0.5 - sx, oy + 0.5 - sy
                d2 = dx * dx + dy * dy
                ins = d2 < rp * rp
                if not ins.any():
                    continue
                dz = np.sqrt(rp * rp - d2[ins]).astype(np.float32)
                k = len(dz)
                px = cx[sel][:, None] + ox[ins][None, :]
                py = cy[sel][:, None] + oy[ins][None, :]
                ok = ((px >= 0) & (px < cam.W) & (py >= 0) & (py < cam.H)).ravel()
                pix_l.append((py * cam.W + px).ravel()[ok])
                z_l.append((Z[sel][:, None] + dz[None, :] / cam.s).ravel()[ok])
                prim_l.append(np.repeat(sel, k)[ok])
                spr_l.append(np.tile(np.arange(k), len(sel))[ok] + n_tab)
                tab.append(np.stack([dx[ins] / rp, -dy[ins] / rp, dz / rp], 1).astype(np.float32))
                n_tab += k
        if pix_l:
            self._submit(np.concatenate(pix_l), np.concatenate(z_l),
                         ("spheres", np.concatenate(prim_l), np.concatenate(tab), np.concatenate(spr_l),
                          np.concatenate(z_l), np.asarray(rgb, np.float32), np.asarray(s.owner, np.int64)))

    def sticks(self, st: Sticks, rgb):
        cam = self.cam
        f32 = np.float32
        ax, ay, az = cam.screen(st.a)
        bx, by, bz = cam.screen(st.b)
        s = f32(cam.s)
        Ax, Ay, Az = ax, f32(cam.H) - ay, az * s  # pixel units, y up
        Dx, Dy, Dz = bx - ax, ay - by, (bz - az) * s
        L = np.sqrt(Dx * Dx + Dy * Dy + Dz * Dz)
        R = np.asarray(st.radius, f32) * s
        ok = (L > 1e-3) & (L * L - Dz * Dz > 1e-4)  # end-on sticks are covered by their balls
        inv = 1 / np.maximum(L, f32(1e-3))
        ux, uy, uz = Dx * inv, Dy * inv, Dz * inv
        a = np.maximum(1 - uz * uz, f32(1e-6))
        by0 = np.clip(np.floor(np.minimum(ay, by) - R), 0, cam.H - 1).astype(np.int64)
        by1 = np.clip(np.ceil(np.maximum(ay, by) + R), 0, cam.H - 1).astype(np.int64)
        sel = np.flatnonzero(ok & (by1 >= by0))
        if not len(sel):
            return
        ddx, ddy = (bx - ax)[sel], (by - ay)[sel]
        L2 = np.sqrt(ddx * ddx + ddy * ddy)
        dx2, dy2 = ddx / np.maximum(L2, 1e-6), ddy / np.maximum(L2, 1e-6)
        nx2, ny2 = -dy2, dx2
        Rs = R[sel] + f32(0.5)
        axs, ays = ax[sel], ay[sel]
        G = np.empty((len(sel), 4, 3), f32)  # screen-space capsule bound
        G[:, 0] = np.stack([nx2, ny2, Rs - nx2 * axs - ny2 * ays], 1)
        G[:, 1] = np.stack([-nx2, -ny2, Rs + nx2 * axs + ny2 * ays], 1)
        G[:, 2] = np.stack([dx2, dy2, Rs - dx2 * axs - dy2 * ays], 1)
        G[:, 3] = np.stack([-dx2, -dy2, L2 + Rs + dx2 * axs + dy2 * ays], 1)
        prim, px, py = _spans(G, by0[sel], by1[sel], cam.W)
        prim = sel[prim]
        p0x = px.astype(f32) + f32(0.5) - Ax[prim]
        p0y = (f32(cam.H - 0.5) - py.astype(f32)) - Ay[prim]
        uxp, uyp, uzp, azp, ap = ux[prim], uy[prim], uz[prim], Az[prim], a[prim]
        m = p0x * uxp + p0y * uyp
        pu = m - azp * uzp
        b = -uzp * m - azp * ap
        c = p0x * p0x + p0y * p0y + azp * azp - pu * pu - R[prim] ** 2
        disc = b * b - ap * c
        tt = (-b + np.sqrt(np.maximum(disc, 0))) / ap
        sp = pu + tt * uzp
        good = np.flatnonzero((disc >= 0) & (sp >= 0) & (sp <= L[prim]))
        prim, px, py, tt, sp = prim[good], px[good], py[good], tt[good], sp[good]
        nrm = np.stack([p0x[good] - sp * uxp[good], p0y[good] - sp * uyp[good],
                        -azp[good] + tt - sp * uzp[good]], 1)
        z = tt / s
        self._submit(py * cam.W + px, z, ("sticks", prim, nrm, z, np.asarray(rgb, np.float32),
                                          np.asarray(st.owner, np.int64)))

    # -- resolve
    def resolve(self):
        """(pixel indices, lit rgb, depth, owner residue) of the covered pixels."""
        win = np.flatnonzero(self.zbuf >= 0)
        key = self.zbuf[win]
        pid = (key >> IDBITS) & 31
        idx = key & ((1 << IDBITS) - 1)
        rgb = np.zeros((len(win), 3), np.float32)
        nrm = np.zeros((len(win), 3), np.float32)
        depth = np.zeros(len(win), np.float32)
        owner = np.full(len(win), -1, np.int64)
        for p, (attrs, offset) in enumerate(self.passes):
            sel = np.flatnonzero(pid == p)
            if not len(sel):
                continue
            k = idx[sel] + offset
            kind = attrs[0]
            if kind == "mesh":
                _, (nx, ny, nz), (t0, t1, t2), prim, l0, l1, z, col, own = attrs
                pk = prim[k]
                v0, v1, v2 = t0[pk], t1[pk], t2[pk]
                a, b = l0[k], l1[k]
                c = 1 - a - b
                n = np.stack([nx[v0] * a + nx[v1] * b + nx[v2] * c, ny[v0] * a + ny[v1] * b + ny[v2] * c,
                              nz[v0] * a + nz[v1] * b + nz[v2] * c], 1)
                # the nearest vertex's colour: crisp colour boundaries
                vert = np.where(a >= b, np.where(a >= c, v0, v2), np.where(b >= c, v1, v2))
                rgb[sel] = col[vert]
                owner[sel] = own[vert]
            elif kind == "spheres":
                _, prim, tab, spr, z, col, own = attrs
                n = tab[spr[k]]
                rgb[sel] = col[prim[k]]
                owner[sel] = own[prim[k]]
            else:
                _, prim, n_all, z, col, own = attrs
                n = n_all[k]
                rgb[sel] = col[prim[k]]
                owner[sel] = own[prim[k]]
            nrm[sel] = n
            depth[sel] = z[k]
        return win, np.clip(shade(nrm, rgb), 0, 1), depth, owner


class SoftwareRenderer:
    """``render(items, viewport, shading, quality)`` -> straight-alpha RGBA, row 0 at the top.

    Stateless (one instance may serve several figures and threads).  ``quality="fast"``
    renders at half resolution without outlines or anti-aliasing (GUI drags)."""

    name = "software"
    draws_glow = True  # the compositor draws the glow halo; the panel must not

    def render(self, items: List[Colored], vp, shading: Shading, quality: str = "high") -> np.ndarray:
        fast = quality == "fast"
        cam = _Cam(vp, items, 0.5 if fast else 1.0)
        opaque = [it for it in items if it.opacity >= 0.999]
        glass = [it for it in items if it.opacity < 0.999]
        pix, rgb, depth, owner = self._draw(cam, opaque).resolve()
        out = composite(cam, pix, rgb, depth, owner, shading, outline=0.0 if fast else 0.5, aa=not fast)
        if glass:
            self._glass(cam, glass, out, pix, depth, shading)
        if fast:  # back to the requested size
            out = np.repeat(np.repeat(out, 2, 0), 2, 1)
            full = np.zeros((vp.height, vp.width, 4), np.uint8)
            h, w = min(vp.height, out.shape[0]), min(vp.width, out.shape[1])
            full[:h, :w] = out[:h, :w]
            out = full
        return out

    @staticmethod
    def _draw(cam: _Cam, items: List[Colored]) -> _ZPass:
        zp = _ZPass(cam)
        for it in items:
            p = it.prim
            if isinstance(p, Mesh) and len(p.tris):
                zp.mesh(p, it.rgb)
            elif isinstance(p, Spheres) and len(p.xyz):
                zp.spheres(p, it.rgb)
            elif isinstance(p, Sticks) and len(p.a):
                zp.sticks(p, it.rgb)
        return zp

    def _glass(self, cam, items, out, opaque_pix, opaque_depth, shading):
        """Translucent representations: the front-most translucent surface in front of the
        opaque one, blended with each representation's opacity."""
        front = np.full(cam.W * cam.H, -np.inf, np.float32)
        front[opaque_pix] = opaque_depth
        flat = out.reshape(-1, 4)
        for it in items:  # one pass per representation (opacity differs)
            pix, rgb, depth, owner = self._draw(cam, [it]).resolve()
            keep = depth > front[pix] + 1e-3
            pix, rgb, depth, owner = pix[keep], rgb[keep], depth[keep], owner[keep]
            if not len(pix):
                continue
            rgb = _tint(_fog(cam, rgb, depth, shading), owner, shading)
            a = np.float32(it.opacity)
            dst_a = flat[pix, 3:4].astype(np.float32) / 255.0
            dst = flat[pix, :3].astype(np.float32) / 255.0
            out_a = a + dst_a * (1 - a)
            col = (rgb * a + dst * dst_a * (1 - a)) / np.maximum(out_a, 1e-6)
            flat[pix, :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
            flat[pix, 3:4] = np.clip(out_a * 255 + 0.5, 0, 255).astype(np.uint8)


def _fog(cam, rgb, depth, shading: Shading):
    """Depth cue toward the panel colour, over the panel's depth range (as the 2D trace)."""
    if shading.fog_strength <= 0:
        return rgb
    span = max(cam.fog_hi - cam.fog_lo, 1e-6)
    f = (np.float32(shading.fog_strength) * (1 - np.clip((depth - cam.fog_lo) / span, 0, 1)))[:, None]
    return rgb * (1 - f) + np.asarray(shading.fog_rgb, np.float32) * f


def _glow_of(owner, shading: Shading) -> np.ndarray:
    g = shading.glow
    if g is None or not len(g):
        return np.zeros(len(owner), np.float32)
    g = np.asarray(g, np.float32)
    ok = (owner >= 0) & (owner < len(g))
    return np.where(ok, g[np.where(ok, owner, 0)], 0).astype(np.float32)


def _tint(rgb, owner, shading: Shading):
    """Glowing residues lighten toward the glow colour (screen blend)."""
    gi = _glow_of(owner, shading)
    if not gi.any():
        return rgb
    t = (np.float32(GLOW_TINT) * gi)[:, None]
    return 1 - (1 - rgb) * (1 - t * np.asarray(shading.glow_rgb, np.float32))


_NORM_CACHE: dict = {}


def _disc_norm(sig: float, rad: float) -> float:
    """1 / peak of a blurred disc of radius ``rad`` (a glowing residue reaches full alpha)."""
    key = (round(sig, 2), round(rad, 2))
    if key not in _NORM_CACHE:
        n = int(np.ceil(3 * sig + rad)) + 2
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        d = ((xx * xx + yy * yy) <= max(rad, 0.5) ** 2).astype(np.float32)
        _NORM_CACHE[key] = 1.0 / max(float(ndimage.gaussian_filter(d, sig, truncate=2.5).max()), 1e-6)
    return _NORM_CACHE[key]


def _upsample(low: np.ndarray, f: int, h: int, w: int) -> np.ndarray:
    """Bilinear upsampling of a coarse grid (values at cell centres) by ``f``, cropped to
    (h, w); edges are clamped."""

    def along0(x):
        pad = np.concatenate([x[:1], x, x[-1:]], 0)
        out = np.empty((x.shape[0] * f,) + x.shape[1:], np.float32)
        for j in range(f):
            o = (j + 0.5) / f - 0.5  # offset of the fine pixel from its coarse centre
            if o < 0:
                out[j::f] = pad[:-2] * np.float32(-o) + pad[1:-1] * np.float32(1 + o)
            else:
                out[j::f] = pad[1:-1] * np.float32(1 - o) + pad[2:] * np.float32(o)
        return out

    return along0(along0(low.astype(np.float32)).T).T[:h, :w]


def _glow_halo(out, cam, pix, gi, shading: Shading, box):
    """A soft halo of the glow colour around glowing pixels (under the molecule)."""
    W, H = cam.W, cam.H
    y0, y1, x0, x1 = box
    f = 4
    pad = int(np.ceil(2.5 * GLOW_RADIUS * cam.s))
    gy0, gx0 = (max(y0 - pad, 0) // f) * f, (max(x0 - pad, 0) // f) * f
    gy1, gx1 = min(y1 + pad, H), min(x1 + pad, W)
    hh, ww = -(-(gy1 - gy0) // f), -(-(gx1 - gx0) // f)
    py, px = pix // W, pix % W
    low = np.zeros(hh * ww, np.float32)
    np.maximum.at(low, ((py - gy0) // f) * ww + (px - gx0) // f, gi)
    sig = GLOW_RADIUS * cam.s / f
    low = ndimage.gaussian_filter(low.reshape(hh, ww), sig, truncate=2.5)
    low *= _disc_norm(sig, GLOW_DISC * cam.s / f)
    a = _upsample(np.clip(low, 0, 1), f, gy1 - gy0, gx1 - gx0)
    reg = out[gy0:gy1, gx0:gx1]
    reg[..., :3] = np.clip(np.asarray(shading.glow_rgb, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)
    reg[..., 3] = np.clip(a * (GLOW_ALPHA * 255) + 0.5, 0, 255).astype(np.uint8)


def composite(cam, pix, rgb, depth, owner, shading: Shading, outline=0.5, outline_depth=1.6,
              aa=True) -> np.ndarray:
    """Covered pixels -> straight-alpha RGBA: fog, glow (tint + halo), ink outlines at depth
    jumps and silhouettes, alpha anti-aliasing of the edges."""
    W, H = cam.W, cam.H
    out = np.zeros((H, W, 4), np.uint8)
    if not len(pix):
        return out
    rgb = _fog(cam, rgb, depth, shading)
    flat = out.reshape(-1, 4)
    py, px = pix // W, pix % W
    y0, y1 = max(int(py.min()) - 2, 0), min(int(py.max()) + 3, H)
    x0, x1 = max(int(px.min()) - 2, 0), min(int(px.max()) + 3, W)
    gi = _glow_of(owner, shading)
    if gi.max(initial=0) > 0.01:
        _glow_halo(out, cam, pix, gi, shading, (y0, y1, x0, x1))
        rgb = _tint(rgb, owner, shading)
    if outline > 0:
        h, w = y1 - y0, x1 - x0
        li = (py - y0) * w + (px - x0)
        dz = np.full(h * w, cam.zmin - 1e3, np.float32)
        dz[li] = depth
        dz = dz.reshape(h, w)
        t = np.float32(outline_depth)
        edge = np.zeros((h, w), bool)
        edge[1:] |= (dz[:-1] - dz[1:]) > t
        edge[:-1] |= (dz[1:] - dz[:-1]) > t
        edge[:, 1:] |= (dz[:, :-1] - dz[:, 1:]) > t
        edge[:, :-1] |= (dz[:, 1:] - dz[:, :-1]) > t
        edge = edge.ravel()
        cre = edge[li]
        rgb = rgb.copy()
        rgb[cre] = rgb[cre] * (1 - outline) + INK * outline
        cover = np.zeros(h * w, bool)
        cover[li] = True
        sil = np.flatnonzero(edge & ~cover)
        sp = (sil // w + y0) * W + (sil % w + x0)
        flat[sp, :3] = np.clip(INK * 255 + 0.5, 0, 255).astype(np.uint8)
        flat[sp, 3] = np.maximum(flat[sp, 3], int(outline * 0.8 * 255))
    flat[pix, :3] = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    flat[pix, 3] = 255
    if aa:
        sub = out[y0:y1, x0:x1, 3]
        a = sub.astype(np.float32)
        a3 = ndimage.uniform_filter(a, 3, mode="constant")
        sub[...] = np.where(np.abs(a3 - a) > 1, 0.5 * (a + a3), a).astype(np.uint8)
    return out
