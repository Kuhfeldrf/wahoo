"""Lifelike wahoo teaser decoy, ~12 in (312 mm overall), printed as two flat halves.

Outputs:
  wahoo_decoy_left.stl / wahoo_decoy_right.stl  - halves, cut face down, no supports
Features: smooth skin (bars are painted, not carved), shallow gill/mouth/lateral-line
detail, 8 mm recess for stick-on lure eyes, 3.5 mm tow-line channel (nose -> belly),
sealed ballast pocket low in the belly, four 1.9 mm holes for 1.75 mm filament pins.
"""
import numpy as np
import manifold3d as m3
from manifold3d import Manifold
import wahoo_model as w

w.L = 300.0
w.N_DORSAL_FINLETS, w.N_ANAL_FINLETS = 9, 9   # wahoo: 8-9 dorsal, ~9 anal finlets
L = w.L
HALF_FIN = 1.2          # each half's share of fin thickness -> 2.4 mm glued

def shallow_grooves(side):
    caps = []
    def chain(pts, r, sink):
        # sphere centres pushed out of the skin so the groove is only `r - sink` deep
        out = []
        for u, ph in pts:
            p = w.surf(u, ph)
            n = np.array([0, np.cos(ph) * w.az(u), np.sin(ph) * w.wid(u)])
            n /= np.linalg.norm(n) + 1e-9
            p = p + n * sink
            p[1] *= side
            out.append(p)
        return w.capsule_chain(out, r, 12)
    # gill-cover edge
    caps += chain([(0.150 + 0.020 * np.cos(ph), ph) for ph in np.radians(np.linspace(62, -72, 14))], 0.7, 0.25)
    # mouth line
    caps += chain([(u, np.radians(-8 - 18 * (u - 0.006) / 0.066)) for u in np.linspace(0.010, 0.072, 9)], 0.55, 0.15)
    # lateral line (high, then dropping steeply under the first dorsal)
    caps += chain([(u, np.radians(52 if u < 0.40 else 52 - 70 * min(1, (u - 0.40) / 0.06)))
                   for u in np.linspace(0.17, 0.86, 34)], 0.4, 0.15)
    return caps

def frame(n):
    """Rotation whose third column is n."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(a, n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n], 1)

def eye_recess(side):
    u, ph = 0.085, np.radians(20)
    p = w.surf(u, ph)
    n = np.array([0, np.cos(ph) * w.az(u), np.sin(ph) * w.wid(u)]); n /= np.linalg.norm(n)
    p[1] *= side; n[1] *= side
    R = frame(n)
    base = p - n * 1.3                                # 1.3 mm deep flat-bottom pocket
    cyl = Manifold.cylinder(8.0, 4.15, 4.15, 64)      # 8.3 mm dia for 8 mm stick-on eyes
    M = np.hstack([R, base[:, None]])
    return cyl.transform(M.tolist())

def pectoral_relief(side):
    """Short sickle-shaped pectoral, held flat against the flank as on a fast-swimming
    wahoo (they tuck into a shallow groove), standing 1.4 mm proud of the skin."""
    u = 0.200
    z0 = w.zc(u) + 0.10 * w.az(u)
    lead = w.bez((u * L, z0 + 3.6), ((u + 0.035) * L, z0 + 4.2), ((u + 0.092) * L, z0 - 1.2), 14)
    trail = w.bez(((u + 0.092) * L, z0 - 1.2), ((u + 0.045) * L, z0 - 1.0), ((u + 0.006) * L, z0 - 3.8), 14)
    p = list(lead) + list(trail[1:])
    ys = w.wid(u + 0.045)
    fin = w.fin_xz(p, ys - 2.2, ys + 1.4)
    # shallow pocket (pectoral groove) around the fin
    groove = w.fin_xz(p, ys - 0.6, ys + 3.0)
    ring = CrossSection_offset_fin(p, 0.9, ys - 0.5, ys + 3.0) - groove
    if side < 0:
        fin, ring = fin.mirror([0, 1, 0]), ring.mirror([0, 1, 0])
    return fin, ring

def CrossSection_offset_fin(p, off, y0, y1):
    from manifold3d import CrossSection, FillRule, JoinType
    pts = np.asarray(p, float)
    cs = CrossSection([pts], FillRule.EvenOdd)
    if cs.area() < 0:
        cs = CrossSection([pts[::-1]], FillRule.EvenOdd)
    cs = cs.offset(off, JoinType.Round, circular_segments=24)
    return cs.extrude(y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])

def pelvic_fins():
    """Pelvic pair under the pectorals; printed joined on the midline so they can't snap."""
    u0 = 0.218
    b = w.bot
    tip = ((u0 + 0.058) * L, b(u0 + 0.04) - 0.026 * L)
    lead = w.bez((u0 * L, b(u0)), ((u0 + 0.02) * L, b(u0 + 0.02) - 0.020 * L), tip, 12)
    trail = w.bez(tip, ((u0 + 0.040) * L, b(u0 + 0.04) - 0.008 * L), ((u0 + 0.036) * L, b(u0 + 0.036) + 0.3), 12)
    base = [(x * L, b(x) + 2.5) for x in np.linspace(u0 + 0.036, u0, 8)]
    return w.fin_xz(list(lead) + list(trail[1:]) + base, -HALF_FIN, HALF_FIN)

def tow_channel():
    r = 1.75
    pts = [(-6.0, 0.0), (0.46 * L, 0.0), (0.50 * L, w.bot(0.50) * 0.45), (0.52 * L, w.bot(0.52) - 3)]
    sph = [Manifold.sphere(r, 24).translate([x, 0, z]) for x, z in pts]
    return Manifold.batch_boolean([Manifold.batch_hull([a, b]) for a, b in zip(sph[:-1], sph[1:])], m3.OpType.Add)

def ballast_pocket():
    x0, x1 = 0.22 * L, 0.40 * L
    zb = w.bot(0.31) + 3.5
    box = Manifold.cube([x1 - x0, 6.0, 6.5]).translate([x0, -3.0, zb])
    return box

PIN_SPOTS = [(0.14, 5.0, 4.0), (0.38, 10.0, 4.0), (0.62, 0.0, 4.0), (0.78, 0.0, 3.0)]

def pin_holes():
    hs = []
    for u, z, d in PIN_SPOTS:
        c = Manifold.cylinder(2 * d, 0.95, 0.95, 24).rotate([90, 0, 0]).translate([u * L, d, z])
        hs.append(c)
    return Manifold.batch_boolean(hs, m3.OpType.Add)

def build_decoy():
    pr, pl = pectoral_relief(1), pectoral_relief(-1)
    core = w.body() + w.keel()
    cuts = shallow_grooves(1) + shallow_grooves(-1) + [eye_recess(1), eye_recess(-1), pr[1], pl[1]]
    core = core - Manifold.batch_boolean(cuts, m3.OpType.Add)
    core = core + pr[0] + pl[0]
    fins = w.median_fins(-HALF_FIN, HALF_FIN) + pelvic_fins()
    fish = core + fins
    fish = fish - tow_channel() - ballast_pocket() - pin_holes()
    return fish

def halves(fish):
    right = fish.trim_by_plane([0, 1, 0], 0.0).rotate([90, 0, 0])      # y>=0 -> z>=0
    left = fish.trim_by_plane([0, -1, 0], 0.0).rotate([-90, 0, 0])     # y<=0 -> z>=0
    out = []
    for h in (left, right):
        bb = h.bounding_box()
        out.append(h.translate([-bb[0], -bb[1], -bb[2]]))
    return out

if __name__ == "__main__":
    import trimesh, time
    t = time.time()
    fish = build_decoy()
    print("built", fish.status(), fish.num_tri(), round(time.time() - t, 1))
    left, right = halves(fish)
    for name, h in (("wahoo_decoy_left.stl", left), ("wahoo_decoy_right.stl", right)):
        tm = w.export(h, name)
        print(name, "watertight", tm.is_watertight, "bodies", len(tm.split(only_watertight=False)),
              "extents", np.round(tm.extents, 1), "vol cm3", round(tm.volume / 1000, 1))
    tm = w.export(fish, "_decoy_assembled.stl")
    print("assembled extents", np.round(tm.extents, 1), "vol", round(tm.volume / 1000, 1))
