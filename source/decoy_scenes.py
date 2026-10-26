import sys, time, numpy as np, trimesh
import wahoo_model as w
w.L = 300.0
import wahoo_decoy as d
from render import Obj, render, look_at, lin, plane

L = w.L
W, H = 1600, 1000
which = sys.argv[1] if len(sys.argv) > 1 else "all"

def tm_of(man):
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties[:, :3], float), m.tri_verts, process=True)

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

EYE_U, EYE_PH = 0.085, np.radians(20)
eye_c = w.surf(EYE_U, EYE_PH)

def paint(p, n):
    x, y, z = p.T
    u = np.clip(x / L, 0, w.X_END)
    top = w._top(u) * L; bot = w._bot(u) * L; wid = w._wid(u) * L
    zc = 0.5 * (top + bot); az = np.maximum(0.5 * (top - bot), 1e-3)
    v = (z - zc) / az
    back, mid = lin("#132c55"), lin("#2f5f9c")
    silver, belly = lin("#aeb9c4"), lin("#e6eaee")
    fin = lin("#18263e")
    c = belly[None] * np.ones((len(p), 1))
    c = c + (silver - belly)[None] * smooth(-0.55, -0.15, v)[:, None]
    c = c + (mid - silver)[None] * smooth(0.15, 0.45, v)[:, None]
    c = c + (back - mid)[None] * smooth(0.45, 0.8, v)[:, None]
    # cobalt tiger bars (painted)
    ph = (x / L - 0.225) / 0.0245 + 0.12 * np.sin(v * 3.1 + x * 0.05)
    fr = np.abs((ph % 1.0) - 0.5)
    bar = smooth(0.27, 0.17, fr) * smooth(-0.62, -0.30, v) * smooth(0.21, 0.25, u) * smooth(0.90, 0.86, u)
    c = c * (1 - 0.85 * bar[:, None]) + lin("#27479a")[None] * 0.85 * bar[:, None]
    # fins: anything outside the body ellipse or behind the peduncle
    r_body = np.sqrt(v ** 2 + (y / np.maximum(wid, 1e-3)) ** 2)
    is_fin = (x > w.X_END * L - 2) | (np.abs(v) > 1.03) | (r_body > 1.04) | (x < 0)
    pect = is_fin & (np.abs(y) > 0.6 * wid) & (u > 0.19) & (u < 0.30)
    c[is_fin] = fin
    c[pect] = lin("#4a6688")
    # stick-on eye
    side = np.sign(y)
    ec = np.stack([np.full_like(x, eye_c[0]), side * eye_c[1], np.full_like(x, eye_c[2])], 1)
    de = np.linalg.norm(p - ec, axis=1)
    near_eye = (de < 4.6) & (np.abs(y) > 2)
    iris = near_eye & (de >= 2.2)
    c[iris] = lin("#8f8466")
    c[near_eye & (de < 2.2)] = lin("#050608")
    return c

if which in ("all", "water"):
    t = time.time()
    fish = d.build_decoy()
    ftm = tm_of(fish)
    fo = Obj(ftm, lin("#888888"), spec=0.55, shine=110, layer=False, color_fn=paint)
    line = trimesh.creation.cylinder(radius=0.45, segment=[[-1.0, 0, 0], [-900, 40, 150]], sections=12)
    lo = Obj(line, lin("#cfd6db"), spec=0.6, shine=80, layer=False, bounds=False)
    cam = look_at((70, -430, 55), (150, 0, 2))
    im = render([fo, lo], cam, 36, W, H, light_dir=(0.15, -0.35, 1.0), fill_dir=(-0.3, -0.9, -0.4),
                key_int=1.9, fill_int=0.55, amb=0.55, sky_dir=(0, -0.3, 1), bg_top="#3c86b5", bg_bot="#0b2440")
    im.save("render_decoy_in_water.png"); print("water", time.time() - t)

if which in ("all", "bed"):
    t = time.time()
    lt = trimesh.load("wahoo_decoy_left.stl")
    rt = trimesh.load("wahoo_decoy_right.stl")
    # flip the right half over (inside face up) so the channel, ballast pocket and pin holes show
    rt.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    rt.apply_translation([0, -rt.bounds[0][1] + 85, -rt.bounds[0][2]])
    lt.apply_translation([0, 0, 0])
    for m in (lt, rt):
        m.apply_translation([-155, -78, 0])
    bed = plane(0, 0, 0, 175, 175, 20)
    pla = lin("#cfcbc2")
    a = Obj(lt, pla, spec=0.25, shine=40)
    b = Obj(rt, pla, spec=0.25, shine=40)
    bd = Obj(bed, lin("#232427"), spec=0.08, shine=10, layer=False, bounds=False)
    cam = look_at((-40, -270, 230), (0, 5, 0))
    im = render([a, b, bd], cam, 40, W, H, light_dir=(-0.5, -0.6, 0.7), fill_dir=(0.6, -0.4, 0.4),
                key_int=1.25, fill_int=0.35, amb=0.35, bg_top="#3a3b3f", bg_bot="#1d1e21")
    im.save("render_decoy_printed_halves.png"); print("bed", time.time() - t)
