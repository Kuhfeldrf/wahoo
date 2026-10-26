"""Procedural 3D-printable wahoo (Acanthocybium solandri).

Outputs (millimetres):
  wahoo_plaque.stl       - flat-back half-relief wall plaque, prints with no supports
  wahoo_full_model.stl   - full-round fish with two sockets in the belly
  wahoo_stand.stl        - display base with two pins that fit the sockets
"""
import numpy as np
import manifold3d as m3
from manifold3d import Manifold, CrossSection, Mesh, FillRule
from scipy.interpolate import PchipInterpolator
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import trimesh

L = 220.0          # body length snout -> fork (mm); total with tail ~ 228 mm
FIN_T = 1.6        # median fin thickness (mm)

# ---------------------------------------------------------------- body outline
# normalised (x/L) control points for dorsal edge, ventral edge, half-width
X  = [0.0, 0.015, 0.04, 0.08, 0.13, 0.20, 0.30, 0.42, 0.55, 0.66, 0.76, 0.84, 0.89, 0.925, 0.945]
TOP= [0.0, 0.007, 0.019, 0.034, 0.048, 0.060, 0.067, 0.068, 0.061, 0.049, 0.034, 0.021, 0.013, 0.0105, 0.0100]
BOT= [0.0,-0.006,-0.016,-0.030,-0.044,-0.057,-0.065,-0.066,-0.059,-0.047,-0.032,-0.019,-0.012,-0.0100,-0.0095]
WID= [0.0, 0.005, 0.014, 0.025, 0.034, 0.040, 0.044, 0.043, 0.038, 0.030, 0.021, 0.013, 0.009, 0.0080, 0.0075]
_top = PchipInterpolator(X, TOP); _bot = PchipInterpolator(X, BOT); _wid = PchipInterpolator(X, WID)
X_END = X[-1]

def top(u):  return float(_top(u)) * L
def bot(u):  return float(_bot(u)) * L
def wid(u):  return float(_wid(u)) * L
def zc(u):   return 0.5 * (top(u) + bot(u))
def az(u):   return 0.5 * (top(u) - bot(u))

def surf(u, phi):
    """Point on the body surface at x=u*L, angle phi (rad; +90deg = dorsal, 0 = +y flank)."""
    return np.array([u * L, wid(u) * np.cos(phi), zc(u) + az(u) * np.sin(phi)])

def to_manifold(verts, faces):
    return Manifold(Mesh(vert_properties=np.asarray(verts, np.float32),
                         tri_verts=np.asarray(faces, np.uint32)))

def body():
    # dense near the snout and tail
    s = np.linspace(0, 1, 260)
    us = X_END * (0.5 - 0.5 * np.cos(np.pi * s))
    us = us[1:]                       # tip vertex added separately
    nth = 96
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    verts = [[0.0, 0.0, 0.0]]
    for u in us:
        for t in th:
            verts.append([u * L, wid(u) * np.cos(t), zc(u) + az(u) * np.sin(t)])
    nr = len(us)
    verts.append([X_END * L + 0.01, 0.0, zc(X_END)])
    faces = []
    ring = lambda r, k: 1 + r * nth + (k % nth)
    for k in range(nth):                               # snout fan
        faces.append([0, ring(0, k + 1), ring(0, k)])
    for r in range(nr - 1):
        for k in range(nth):
            a, b = ring(r, k), ring(r, k + 1)
            c, d = ring(r + 1, k), ring(r + 1, k + 1)
            faces += [[a, b, d], [a, d, c]]
    end = len(verts) - 1
    for k in range(nth):
        faces.append([end, ring(nr - 1, k), ring(nr - 1, k + 1)])
    tm = trimesh.Trimesh(verts, faces, process=True)
    if tm.volume < 0:
        tm.invert()
    return to_manifold(tm.vertices, tm.faces)

# ---------------------------------------------------------------- 2D helpers
def bez(p0, p1, p2, n=12):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)

def fin_xz(pts, y0, y1):
    """Polygon in the (x,z) plane, extruded across y in [y0, y1]."""
    pts = np.asarray(pts, float)
    cs = CrossSection([pts], FillRule.EvenOdd)
    if cs.area() < 0:
        cs = CrossSection([pts[::-1]], FillRule.EvenOdd)
    solid = cs.extrude(y1 - y0)                 # polygon in XY, thickness along Z
    solid = solid.rotate([90, 0, 0])            # (x,y,z) -> (x,-z,y): Y->Z, Z->-Y
    return solid.translate([0, y1, 0])          # occupies y in [y0, y1]

def base_line(u0, u1, depth, which="top", n=30):
    us = np.linspace(u0, u1, n)
    if which == "top":
        return [(u * L, top(u) - depth) for u in us]
    return [(u * L, bot(u) + depth) for u in us]

# ---------------------------------------------------------------- median fins (x,z polygons)
def first_dorsal():
    ua, ub, n = 0.215, 0.525, 23
    def h(u):
        f = (u - ua) / (ub - ua)
        return L * (0.040 * (1 - f) + 0.013 * f) * min(1, (u - ua) / 0.012 + 0.25)
    tips = np.linspace(ua + 0.004, ub - 0.006, n)
    edge = [(ua * L, top(ua) + 0.2)]
    for i, u in enumerate(tips):
        rake = 0.0045
        edge.append(((u + rake) * L, top(u) + h(u)))
        if i < n - 1:
            um = 0.5 * (u + tips[i + 1]) + rake * 0.6
            edge.append((um * L, top(um) + 0.62 * h(um)))
    edge.append(((ub + 0.004) * L, top(ub) + 0.1))
    back = base_line(ub + 0.004, ua, 2.5, "top")
    return edge + back

def second_dorsal(sign=1, u0=0.552):
    f = (lambda u: top(u)) if sign > 0 else (lambda u: bot(u))
    t0, t1 = u0, u0 + 0.058
    tip = ((u0 + 0.030) * L, f(u0 + 0.02) + sign * 0.034 * L)
    lead = bez((t0 * L, f(t0)), ((t0 + 0.012) * L, f(t0) + sign * 0.022 * L), tip)
    trail = bez(tip, ((u0 + 0.034) * L, f(u0 + 0.04) + sign * 0.010 * L), (t1 * L, f(t1) + sign * 0.6))
    base = [(u * L, f(u) - sign * 2.5) for u in np.linspace(t1, t0, 15)]
    return list(lead) + list(trail[1:]) + base

def finlets(sign=1, u0=0.632, u1=0.872, n=8):
    f = (lambda u: top(u)) if sign > 0 else (lambda u: bot(u))
    polys = []
    for i, u in enumerate(np.linspace(u0, u1, n)):
        hh = L * (0.016 - 0.005 * i / (n - 1))
        b = 0.016
        tip = ((u + b + 0.010) * L, f(u + b * 0.6) + sign * hh)
        p = [(u * L, f(u) - sign * 1.5), (u * L, f(u) + sign * 0.3),
             ((u + 0.004) * L, f(u) + sign * hh * 0.55), tip,
             ((u + b) * L, f(u + b) + sign * 0.3), ((u + b) * L, f(u + b) - sign * 1.5)]
        polys.append(p)
    return polys

def caudal():
    r0, r1 = 0.895, 0.950
    zr = zc(0.92)
    up_tip = (1.040 * L, zr + 0.122 * L)
    lo_tip = (1.040 * L, zr - 0.118 * L)
    fork = (0.972 * L, zr)
    a = bez((r0 * L, zr + 0.6 * az(r0)), (0.975 * L, zr + 0.035 * L), up_tip, 18)
    b = bez(up_tip, (1.000 * L, zr + 0.050 * L), fork, 18)
    c = bez(fork, (1.000 * L, zr - 0.048 * L), lo_tip, 18)
    d = bez(lo_tip, (0.975 * L, zr - 0.034 * L), (r0 * L, zr - 0.6 * az(r0)), 18)
    return list(a) + list(b[1:]) + list(c[1:]) + list(d[1:-1])

N_DORSAL_FINLETS, N_ANAL_FINLETS = 8, 8

def median_fins(y0, y1):
    parts = [fin_xz(first_dorsal(), y0, y1),
             fin_xz(second_dorsal(+1, 0.552), y0, y1),
             fin_xz(second_dorsal(-1, 0.566), y0, y1),
             fin_xz(caudal(), y0, y1)]
    parts += [fin_xz(p, y0, y1) for p in finlets(+1, 0.628, 0.872, N_DORSAL_FINLETS)]
    parts += [fin_xz(p, y0, y1) for p in finlets(-1, 0.640, 0.872, N_ANAL_FINLETS)]
    return Manifold.batch_boolean(parts, m3.OpType.Add)

# ---------------------------------------------------------------- paired features
def keel():
    """Horizontal caudal keel on the peduncle (both sides), polygon in x-y."""
    us = np.linspace(0.855, 0.935, 20)
    out = [(u * L, wid(u) + 1.6 * np.sin(np.pi * (u - 0.855) / 0.08) + 0.2) for u in us]
    pts = [(0.855 * L, 0.0)] + out + [(0.935 * L, 0.0)]
    right = CrossSection([np.array(pts)], FillRule.EvenOdd)
    if right.area() < 0:
        right = CrossSection([np.array(pts[::-1])], FillRule.EvenOdd)
    both = right + right.mirror([0, 1])
    return both.extrude(1.6).translate([0, 0, zc(0.9) - 0.8])

def pectoral(side=1):
    u = 0.205
    z0 = zc(u) + 0.10 * az(u)
    p = [(u * L, z0 + 2.8), ((u + 0.02) * L, z0 + 2.7), ((u + 0.055) * L, z0 + 0.6),
         ((u + 0.088) * L, z0 - 2.4), ((u + 0.05) * L, z0 - 2.9), ((u + 0.003) * L, z0 - 2.7)]
    fin = fin_xz(p, -0.65, 0.65)
    # tilt outward (rotate about x through root), then sit it on the flank
    fin = fin.translate([0, 0, -z0]).rotate([-8 * side, 0, 0]).translate([0, 0, z0])
    y_surf = wid(u + 0.03) * np.sqrt(max(0.0, 1 - (0.10) ** 2))
    fin = fin.translate([0, side * (y_surf - 0.9), 0])
    return fin

def capsule_chain(pts, r, seg=10):
    sph = [Manifold.sphere(r, seg).translate(list(p)) for p in pts]
    return [Manifold.batch_hull([a, b]) for a, b in zip(sph[:-1], sph[1:])]

def grooves(side=1):
    s = side
    caps = []
    # tiger bars (~26 vertical bars on the flank)
    for i, u in enumerate(np.linspace(0.245, 0.835, 26)):
        lo = -0.30 if i % 3 else -0.45
        phis = np.radians(np.linspace(80, lo * 90, 8))
        pts = []
        for j, ph in enumerate(phis):
            uu = u + 0.010 * (j / 7) - 0.004 * np.sin(np.pi * j / 7)
            p = surf(uu, ph); p[1] *= s
            pts.append(p)
        caps += capsule_chain(pts, 0.55)
    # gill-cover edge
    pts = []
    for ph in np.radians(np.linspace(62, -72, 12)):
        uu = 0.150 + 0.020 * np.cos(ph)
        p = surf(uu, ph); p[1] *= s; pts.append(p)
    caps += capsule_chain(pts, 0.6)
    # mouth line
    pts = []
    for uu in np.linspace(0.006, 0.072, 9):
        ph = np.radians(-8 - 18 * (uu - 0.006) / 0.066)
        p = surf(uu, ph); p[1] *= s; pts.append(p)
    caps += capsule_chain(pts, 0.45)
    # lateral line: high then dropping sharply under the first dorsal (as in wahoo)
    pts = []
    for uu in np.linspace(0.17, 0.86, 30):
        ph = np.radians(52 if uu < 0.40 else 52 - 70 * min(1, (uu - 0.40) / 0.06))
        p = surf(uu, ph); p[1] *= s; pts.append(p)
    caps += capsule_chain(pts, 0.35, 8)
    return caps

def eye_parts(side=1):
    u, ph = 0.085, np.radians(20)
    p = surf(u, ph)
    n = np.array([0, np.cos(ph) * az(u), np.sin(ph) * wid(u)]); n /= np.linalg.norm(n)
    p[1] *= side; n[1] *= side
    socket = Manifold.sphere(3.6, 40).translate(list(p + 2.4 * n))
    ball = Manifold.sphere(2.4, 40).translate(list(p - 1.0 * n))
    pupil = Manifold.sphere(0.9, 20).translate(list(p + (1.4 + 0.5) * n))
    return socket, ball, pupil

def fish_core(sides=(1, -1)):
    core = body() + keel()
    for s in sides:
        core = core + pectoral(s)
    cuts, adds, pupils = [], [], []
    for s in sides:
        cuts += grooves(s)
        sk, ball, pu = eye_parts(s)
        cuts.append(sk); adds.append(ball); pupils.append(pu)
    core = core - Manifold.batch_boolean(cuts, m3.OpType.Add)
    core = core + Manifold.batch_boolean(adds, m3.OpType.Add)
    core = core - Manifold.batch_boolean(pupils, m3.OpType.Add)
    return core

# ---------------------------------------------------------------- text
def text_cs(s, size, family="DejaVu Sans", weight="bold", style="normal", spacing=0.0):
    fp = FontProperties(family=family, weight=weight, style=style)
    x = 0.0
    polys = []
    for ch in s:
        if ch == " ":
            x += size * 0.35; continue
        tp = TextPath((x, 0), ch, size=size, prop=fp)
        polys += [np.array(p) for p in tp.to_polygons() if len(p) > 2]
        x = max(p[:, 0].max() for p in polys) + spacing * size
    cs = CrossSection(polys, FillRule.EvenOdd)
    (x0, y0, x1, y1) = cs.bounds()
    return cs.translate([-(x0 + x1) / 2, -(y0 + y1) / 2])

def rounded_rect(w, h, r):
    return CrossSection.square([w - 2 * r, h - 2 * r], True).offset(r, m3.JoinType.Round, circular_segments=64)

# ---------------------------------------------------------------- builds
def build_full():
    fish = fish_core((1, -1)) + median_fins(-FIN_T / 2, FIN_T / 2)
    holes = []
    for u in (0.30, 0.47):
        holes.append(Manifold.cylinder(14, 3.2, 3.2, 48).translate([u * L, 0, bot(u) - 2]))
    fish = fish - Manifold.batch_boolean(holes, m3.OpType.Add)
    return fish

def build_stand(fish):
    lift = 34.0                        # gap between base top and lowest belly point
    zmin = fish.bounding_box()[2]
    base_t = 5.0
    base = CrossSection.circle(1, 128).scale([72, 25]).extrude(base_t)
    # chamfer-ish top: slightly smaller second layer
    top_l = CrossSection.circle(1, 128).scale([69.5, 22.5]).extrude(1.2).translate([0, 0, base_t - 0.2])
    xm = 0.385 * L
    st = (base + top_l).translate([xm, 0, 0])
    ztop = base_t + 1.0
    pins = []
    for u in (0.30, 0.47):
        z_belly = bot(u) - zmin + ztop + lift       # belly height in stand coords
        h = z_belly - ztop + 8.5                     # 8.5 mm into the 12 mm socket
        pin = Manifold.cylinder(h, 3.0, 3.0, 48).translate([u * L, 0, ztop])
        collar = Manifold.cylinder(4, 5.5, 3.0, 48).translate([u * L, 0, ztop])
        pins += [pin, collar]
    label = text_cs("WAHOO", 8.5, spacing=0.12).extrude(0.8).translate([xm, -13.5, ztop - 0.2])
    sci = text_cs("Acanthocybium solandri", 4.6, weight="bold", style="oblique").extrude(0.8).translate([xm, 14.0, ztop - 0.2])
    stand = st + Manifold.batch_boolean(pins, m3.OpType.Add) + label + sci
    return stand, ztop + lift - zmin

def keyhole(cx, cy):
    """Cut from the back face (z=0 up): nail head enters the round hole and slides up the slot."""
    narrow = (CrossSection.circle(5.0, 48).translate([0, -7]) +
              CrossSection.square([4.2, 8], True)).extrude(1.3)
    wide = (CrossSection.circle(5.0, 48).translate([0, -7]) +
            CrossSection.circle(5.0, 48).translate([0, 1]) +
            CrossSection.square([10, 8], True).translate([0, -3])).extrude(1.8).translate([0, 0, 1.3])
    return (narrow + wide).translate([cx, cy, 0])

def build_plaque():
    """Cut-out silhouette plaque: half-relief fish on a backing that follows its outline,
    with a name bar under the belly and two keyhole hangers in the back."""
    half = fish_core((1,)).trim_by_plane([0, 1, 0], 0.0)
    relief = half + median_fins(0.0, 1.8)
    # (x, y, z) -> (x, z, y): flank faces +Z, dorsal toward +Y
    relief = relief.rotate([90, 0, 0]).mirror([0, 1, 0])
    bb = relief.bounding_box()
    T = 4.5
    relief = relief.translate([-(bb[0] + bb[3]) / 2, -zc(0.4), T - 0.4])
    foot = relief.project()
    # close the gaps between finlets/fins, then pad the outline
    outline = foot.offset(7.0, m3.JoinType.Round, circular_segments=48).offset(-2.5, m3.JoinType.Round, circular_segments=48)
    x_bar = (0.40 - 0.5) * L
    y_bar = bot(0.40) - zc(0.4) - 15.0
    bar = rounded_rect(150, 32, 10).translate([x_bar, y_bar])
    back = (outline + bar).simplify(0.08)
    back = CrossSection([max(back.to_polygons(), key=lambda p: abs(CrossSection([p]).area()))])
    lower = back.extrude(T - 1.0)
    upper = back.offset(-1.2, m3.JoinType.Round, circular_segments=48).simplify(0.05).extrude(1.3).translate([0, 0, T - 1.3])
    title = text_cs("WAHOO", 11.0, spacing=0.18).extrude(1.2).translate([x_bar, y_bar + 5.0, T - 0.2])
    sci = text_cs("Acanthocybium solandri", 5.2, weight="bold", style="oblique").extrude(1.0).translate([x_bar, y_bar - 8.5, T - 0.2])
    plq = lower + upper + relief + title + sci
    yk = 2.0
    plq = plq - keyhole((0.30 - 0.5) * L, yk) - keyhole((0.54 - 0.5) * L, yk)
    return plq

def export(man, path):
    man = man.simplify(0.002)
    mesh = man.to_mesh()
    tm = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=True)
    tm.export(path)
    return tm

if __name__ == "__main__":
    import sys, time
    t = time.time()
    full = build_full()
    print("full", full.status(), full.num_tri(), time.time() - t)
    stand, z_off = build_stand(full)
    full_on_stand = full.translate([0, 0, z_off])
    tf = export(full_on_stand.translate([0, 0, -full_on_stand.bounding_box()[2]]), "wahoo_full_model.stl")
    ts = export(stand, "wahoo_stand.stl")
    export(full_on_stand + stand, "_assembled_preview.stl")
    print("stand", stand.status(), time.time() - t)
    plq = build_plaque()
    tp = export(plq, "wahoo_plaque.stl")
    print("plaque", plq.status(), plq.num_tri(), time.time() - t)
    for name, tm in [("full", tf), ("stand", ts), ("plaque", tp)]:
        print(name, "watertight", tm.is_watertight, "extents", np.round(tm.extents, 1),
              "vol cm3", round(tm.volume / 1000, 1), "bodies", len(tm.split(only_watertight=False)))
