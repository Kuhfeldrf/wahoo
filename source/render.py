"""Tiny numpy software renderer: z-buffer, smooth normals, shadow map, soft contact
occlusion, faint 0.2 mm layer lines - enough to preview how a PLA print will look."""
import numpy as np, trimesh
from PIL import Image

# ------------------------------------------------------------------ rasterizer
def raster(P, F, W, H):
    """P: (Nv,3) pixel x, pixel y, depth (smaller = nearer). Returns depth, tri id, b1, b2."""
    T = P[F]
    x, y, z = T[..., 0], T[..., 1], T[..., 2]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    xmin = np.floor(x.min(1) - 0.5).astype(int); xmax = np.ceil(x.max(1) - 0.5).astype(int)
    ymin = np.floor(y.min(1) - 0.5).astype(int); ymax = np.ceil(y.max(1) - 0.5).astype(int)
    xmin = np.clip(xmin, 0, W - 1); xmax = np.clip(xmax, 0, W - 1)
    ymin = np.clip(ymin, 0, H - 1); ymax = np.clip(ymax, 0, H - 1)
    ok = (np.abs(area) > 1e-12) & (x.max(1) >= 0) & (x.min(1) < W) & (y.max(1) >= 0) & (y.min(1) < H) & (z.min(1) > 1e-3)
    size = np.maximum(xmax - xmin, ymax - ymin) + 1
    frags = []

    def emit(ids, px, py):  # px,py shape (n,k)
        X0, Y0 = x[ids][:, None, :], y[ids][:, None, :]
        cx, cy = px + 0.5, py + 0.5
        a = area[ids][:, None]
        w0 = ((X0[..., 1] - cx) * (Y0[..., 2] - cy) - (X0[..., 2] - cx) * (Y0[..., 1] - cy)) / a
        w1 = ((X0[..., 2] - cx) * (Y0[..., 0] - cy) - (X0[..., 0] - cx) * (Y0[..., 2] - cy)) / a
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (px <= xmax[ids][:, None]) & (py <= ymax[ids][:, None])
        Z = z[ids][:, None, :]
        d = w0 * Z[..., 0] + w1 * Z[..., 1] + w2 * Z[..., 2]
        tid = np.broadcast_to(ids[:, None], px.shape)
        frags.append(((py * W + px)[m], d[m], tid[m], w1[m], w2[m]))

    for K in (2, 4, 8, 16, 32, 64):
        sel = np.where(ok & (size <= K) & (size > K // 2 if K > 2 else True))[0]
        if not len(sel):
            continue
        g = np.arange(K * K)
        gx, gy = g % K, g // K
        step = max(1, 3_000_000 // (K * K))
        for i in range(0, len(sel), step):
            ids = sel[i:i + step]
            emit(ids, xmin[ids][:, None] + gx[None], ymin[ids][:, None] + gy[None])
    big = np.where(ok & (size > 64))[0]
    for t in big:
        gx, gy = np.meshgrid(np.arange(xmin[t], xmax[t] + 1), np.arange(ymin[t], ymax[t] + 1))
        gx, gy = gx.ravel(), gy.ravel()
        for j in range(0, len(gx), 4_000_000):
            emit(np.array([t]), gx[None, j:j + 4_000_000], gy[None, j:j + 4_000_000])
    pix = np.concatenate([f[0] for f in frags]); dep = np.concatenate([f[1] for f in frags])
    tid = np.concatenate([f[2] for f in frags]); b1 = np.concatenate([f[3] for f in frags]); b2 = np.concatenate([f[4] for f in frags])
    keep = dep > 0
    pix, dep, tid, b1, b2 = pix[keep], dep[keep], tid[keep], b1[keep], b2[keep]
    order = np.lexsort((dep, pix))
    pix, dep, tid, b1, b2 = pix[order], dep[order], tid[order], b1[order], b2[order]
    first = np.ones(len(pix), bool); first[1:] = pix[1:] != pix[:-1]
    D = np.full(W * H, np.inf); I = np.full(W * H, -1); B1 = np.zeros(W * H); B2 = np.zeros(W * H)
    D[pix[first]] = dep[first]; I[pix[first]] = tid[first]; B1[pix[first]] = b1[first]; B2[pix[first]] = b2[first]
    return D.reshape(H, W), I.reshape(H, W), B1.reshape(H, W), B2.reshape(H, W)

# ------------------------------------------------------------------ cameras
def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f

def persp(Vw, cam, fov_deg, W, H):
    eye, r, u, f = cam
    d = Vw - eye
    cx, cy, cz = d @ r, d @ u, d @ f
    s = 0.5 * H / np.tan(np.radians(fov_deg) / 2)
    return np.stack([W / 2 + s * cx / cz, H / 2 - s * cy / cz, cz], 1)

def ortho(Vw, cam, scale, W, H, center2d=(0, 0)):
    eye, r, u, f = cam
    d = Vw - eye
    return np.stack([W / 2 + scale * (d @ r - center2d[0]), H / 2 - scale * (d @ u - center2d[1]), d @ f], 1)

# ------------------------------------------------------------------ scene
class Obj:
    def __init__(self, mesh, color, print_z=None, spec=0.18, shine=40, layer=True, bounds=True, color_fn=None):
        m = trimesh.graph.smooth_shade(mesh, angle=np.radians(32))
        self.V = np.asarray(m.vertices, float); self.F = np.asarray(m.faces)
        self.N = np.asarray(m.vertex_normals, float)
        self.color = np.asarray(color, float); self.spec = spec; self.shine = shine
        self.pz = (self.V[:, 2] if print_z is None else print_z(self.V)); self.layer = layer; self.bounds = bounds; self.color_fn = color_fn

def plane(cx, cy, cz, sx, sy, n=60):
    xs = np.linspace(cx - sx, cx + sx, n + 1); ys = np.linspace(cy - sy, cy + sy, n + 1)
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X.ravel(), Y.ravel(), np.full(X.size, cz)], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return trimesh.Trimesh(V, F, process=False)

def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

def lin(hexcol):
    h = hexcol.lstrip('#'); c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def render(objs, cam, fov, W, H, light_dir, ss=2, bg_top="#e9edf1", bg_bot="#c9d0d8",
           fill_dir=(0.5, -0.8, 0.3), key_int=2.6, fill_int=0.55, amb=0.38, sky_dir=(0, 0, 1),
           floor=None, fog=None):
    Ws, Hs = W * ss, H * ss
    V = np.concatenate([o.V for o in objs]); N = np.concatenate([o.N for o in objs])
    PZ = np.concatenate([o.pz for o in objs])
    Vb = np.concatenate([o.V for o in objs if o.bounds])
    offs = np.cumsum([0] + [len(o.V) for o in objs])
    F = np.concatenate([o.F + offs[i] for i, o in enumerate(objs)])
    mat = np.concatenate([np.full(len(o.F), i) for i, o in enumerate(objs)])
    P = persp(V, cam, fov, Ws, Hs)
    D, I, B1, B2 = raster(P, F, Ws, Hs)
    hit = I >= 0
    t = I[hit]; b1 = B1[hit]; b2 = B2[hit]; b0 = 1 - b1 - b2
    f = F[t]
    # perspective-correct barycentrics
    iz = 1.0 / P[:, 2]
    c0, c1, c2 = b0 * iz[f[:, 0]], b1 * iz[f[:, 1]], b2 * iz[f[:, 2]]
    sm = c0 + c1 + c2
    b0, b1, b2 = c0 / sm, c1 / sm, c2 / sm
    pos = b0[:, None] * V[f[:, 0]] + b1[:, None] * V[f[:, 1]] + b2[:, None] * V[f[:, 2]]
    nrm = b0[:, None] * N[f[:, 0]] + b1[:, None] * N[f[:, 1]] + b2[:, None] * N[f[:, 2]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    pz = b0 * PZ[f[:, 0]] + b1 * PZ[f[:, 1]] + b2 * PZ[f[:, 2]]
    eye = cam[0]
    view = eye - pos; view /= np.linalg.norm(view, axis=1, keepdims=True)
    flip = (nrm * view).sum(1) < 0
    nrm[flip] *= -1

    def shadow_vis(ldir, res, pcf, bias, spread, nofs, pad):
        ldir = np.asarray(ldir, float); ldir /= np.linalg.norm(ldir)
        ctr = V.mean(0)
        _, r, u, fw = look_at(ctr + ldir * 2000, ctr, up=(0, 1, 0) if abs(ldir[2]) > 0.9 else (0, 0, 1))
        to_l = lambda X: np.stack([(X - ctr) @ r, -((X - ctr) @ u), (X - ctr) @ fw], 1)
        A = to_l(V)
        Ab = to_l(Vb)
        lo = Ab.min(0) - pad; span = max(np.ptp(Ab[:, 0]), np.ptp(Ab[:, 1])) + 2 * pad
        lo[2] = A[:, 2].min()
        sc = (res - 4) / span
        tf = lambda Q: np.stack([(Q[:, 0] - lo[0]) * sc + 2, (Q[:, 1] - lo[1]) * sc + 2, Q[:, 2] - lo[2] + 1.0], 1)
        Dl, _, _, _ = raster(tf(A), F, res, res)
        q = tf(to_l(pos + nrm * nofs))
        inside = (q[:, 0] >= 1) & (q[:, 0] < res - 1) & (q[:, 1] >= 1) & (q[:, 1] < res - 1)
        vis = np.zeros(len(q)); cnt = 0
        rng = np.arange(-pcf, pcf + 1) * spread
        for dx in rng:
            for dy in rng:
                ix = np.clip((q[:, 0] - 0.5 + dx).round().astype(int), 0, res - 1)
                iy = np.clip((q[:, 1] - 0.5 + dy).round().astype(int), 0, res - 1)
                vis += (q[:, 2] <= Dl[iy, ix] + bias); cnt += 1
        vis = vis / cnt
        vis[~inside] = 1.0
        return vis, ldir

    key_vis, L = shadow_vis(light_dir, 4096, 2, 0.25, 1, 0.35, 20)
    sky_vis, S = shadow_vis(sky_dir, 1024, 3, 1.0, 3, 0.8, 60)
    Fd = np.asarray(fill_dir, float); Fd /= np.linalg.norm(Fd)

    col = np.zeros((len(t), 3))
    m = mat[t]
    for i, o in enumerate(objs):
        sel = m == i
        if not sel.any():
            continue
        n = nrm[sel]; v = view[sel]; p = pos[sel]
        alb = np.broadcast_to(o.color, (sel.sum(), 3)).copy() if o.color_fn is None else o.color_fn(p, n)
        sp_w = np.ones(sel.sum()) if o.color_fn is None or not hasattr(o, 'spec_fn') else o.spec_fn(p)
        if o.layer:
            side = np.sqrt(np.clip(1 - n[:, 2] ** 2, 0, 1))
            lines = 0.5 + 0.5 * np.sin(2 * np.pi * pz[sel] / 0.2)
            alb *= (1 - 0.07 * side * lines)[:, None]
        ndl = np.clip(n @ L, 0, None)
        h = L + v; h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec = sp_w * o.spec * np.clip((n * h).sum(1), 0, None) ** o.shine * (ndl > 0)
        fill = np.clip(n @ Fd, 0, None)
        hemi = 0.6 + 0.4 * n[:, 2]
        fres = (1 - np.clip((n * v).sum(1), 0, 1)) ** 4
        lit = alb * (key_int * ndl * key_vis[sel])[:, None] + alb * (fill_int * fill)[:, None] \
            + alb * (amb * hemi * (0.35 + 0.65 * sky_vis[sel]))[:, None] \
            + (spec * key_vis[sel] * key_int)[:, None] + 0.10 * fres[:, None] * sky_vis[sel][:, None]
        col[sel] = lit
    if fog is not None:
        dist = np.linalg.norm(pos - eye, axis=1)
        k = np.clip((dist - fog[0]) / (fog[1] - fog[0]), 0, 1)[:, None]
    img = np.zeros((Hs, Ws, 3))
    gy = np.linspace(0, 1, Hs)[:, None, None]
    img[:] = (1 - gy) * lin(bg_top) + gy * lin(bg_bot)
    flat = img.reshape(-1, 3)
    idx = np.where(hit.ravel())[0]
    if fog is not None:
        col = col * (1 - k) + flat[idx] * k
    flat[idx] = col
    img = flat.reshape(Hs, Ws, 3)
    # simple filmic-ish tone curve
    img = img / (1 + 0.12 * img)
    img = srgb(img)
    im = Image.fromarray((img * 255).astype(np.uint8))
    return im.resize((W, H), Image.LANCZOS)
