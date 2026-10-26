import numpy as np, trimesh, struct
import wahoo_model as w
w.L = 300.0
import wahoo_decoy as d
import sys; sys.argv = ['x', 'none']
import decoy_scenes as ds   # paint()
L = w.L

def pack(path, V, N, C, F, LAB=None):
    V = np.asarray(V, np.float32); F = np.asarray(F, np.uint32)
    Nq = np.clip(np.round(np.asarray(N) * 127), -127, 127).astype(np.int8)
    C = np.asarray(C, np.uint8)
    LAB = np.zeros(len(V), np.uint8) if LAB is None else np.asarray(LAB, np.uint8)
    body = V.tobytes() + Nq.tobytes() + C.tobytes() + LAB.tobytes()
    body += b"\0" * ((-len(body)) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(V), len(F)) + body + F.tobytes())

def to3(X):  # model (x,y,z) -> three (x, z, -y)
    return np.stack([X[:, 0], X[:, 2], -X[:, 1]], 1)

def srgb8(c):
    c = np.clip(c, 0, 1); c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return (c * 255).round()

fish = d.build_decoy(); m = fish.to_mesh()
tm = trimesh.Trimesh(m.vert_properties[:, :3].astype(float), m.tri_verts, process=True)
tm = trimesh.graph.smooth_shade(tm, angle=np.radians(32))
V = np.asarray(tm.vertices); N = np.asarray(tm.vertex_normals)
col = ds.paint(V, N)
# fin labels
x, y, z = V.T; u = x / L; uc = np.clip(u, 0, w.X_END)
top = w._top(uc) * L; bot = w._bot(uc) * L; wid = w._wid(uc) * L
zc = (top + bot) / 2; az = np.maximum((top - bot) / 2, 1e-3); v = (z - zc) / az
rb = np.sqrt(v ** 2 + (y / np.maximum(wid, 1e-3)) ** 2)
def labels(P):
    x, y, z = P.T; u = x / L; uc = np.clip(u, 0, w.X_END)
    top = w._top(uc) * L; bot = w._bot(uc) * L; wid = w._wid(uc) * L
    zc = (top + bot) / 2; az = np.maximum((top - bot) / 2, 1e-3); v = (z - zc) / az
    rb = np.sqrt(v ** 2 + (y / np.maximum(wid, 1e-3)) ** 2)
    lab = np.zeros(len(P), np.uint8)
    up, dn = v > 1.0, v < -1.0
    lab[up & (u < 0.535)] = 1; lab[up & (u >= 0.535) & (u < 0.622)] = 2; lab[up & (u >= 0.622) & (u < 0.94)] = 3
    lab[dn & (u < 0.30)] = 7; lab[dn & (u >= 0.53) & (u < 0.634)] = 5; lab[dn & (u >= 0.634) & (u < 0.94)] = 6
    lab[(rb > 1.03) & (np.abs(v) < 0.75) & (u > 0.185) & (u < 0.31)] = 8
    lab[(rb > 1.03) & (np.abs(v) < 0.6) & (u > 0.85) & (u < 0.945)] = 9
    lab[u >= 0.945] = 4
    return lab
flab = labels(V[tm.faces].mean(1))
lab = np.zeros(len(V), np.uint8)
for k in range(3):
    np.maximum.at(lab, tm.faces[:, k], flab)
fincol = col.copy()
col[(lab > 0) & (lab != 8)] = ds.lin("#18263e")
col[lab == 8] = ds.lin("#4a6688")
print({i: int((lab == i).sum()) for i in range(10)})
X3 = to3(V); X3[:, 0] -= 150
pack("viewer/decoy.bin", X3, to3(N), srgb8(col), tm.faces, lab)

# stick-on eyes: flattened domes in the recesses
eyes_V, eyes_N, eyes_C, eyes_F = [], [], [], []
off = 0
for side in (1, -1):
    uu, ph = 0.085, np.radians(20)
    p = w.surf(uu, ph); n = np.array([0, np.cos(ph) * w.az(uu), np.sin(ph) * w.wid(uu)]); n /= np.linalg.norm(n)
    p[1] *= side; n[1] *= side
    R = d.frame(n)
    s = trimesh.creation.icosphere(4, 1.0)
    sv = s.vertices[s.vertices[:, 2] > -0.05].copy()
    keep = s.vertices[:, 2] > -0.05
    idx = -np.ones(len(s.vertices), int); idx[keep] = np.arange(keep.sum())
    fk = keep[s.faces].all(1); faces = idx[s.faces[fk]]
    loc = np.stack([sv[:, 0] * 4.0, sv[:, 1] * 4.0, sv[:, 2] * 1.6], 1)
    nl = np.stack([sv[:, 0] / 4.0, sv[:, 1] / 4.0, sv[:, 2] / 1.6], 1); nl /= np.linalg.norm(nl, axis=1, keepdims=True)
    world = (R @ loc.T).T + (p - n * 1.1); wn = (R @ nl.T).T
    rr = np.hypot(sv[:, 0], sv[:, 1]) * 4.0
    c = np.where(rr[:, None] < 2.0, [8, 9, 12], np.where(rr[:, None] < 3.6, [196, 168, 92], [40, 38, 30]))
    eyes_V.append(world); eyes_N.append(wn); eyes_C.append(c); eyes_F.append(faces + off); off += len(world)
EV = to3(np.concatenate(eyes_V)); EV[:, 0] -= 150
pack("viewer/eyes.bin", EV, to3(np.concatenate(eyes_N)), np.concatenate(eyes_C), np.concatenate(eyes_F))

# printed halves (print orientation), right half flipped to show the inside
for name, flip, dy in (("left", False, 0.0), ("right", True, 85.0)):
    h = trimesh.load(f"wahoo_decoy_{name}.stl")
    if flip:
        h.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    h.apply_translation([-h.bounds[0][0] - 154, -h.bounds[0][1] + dy - 78, -h.bounds[0][2]])
    h = trimesh.graph.smooth_shade(h, angle=np.radians(32))
    pack(f"viewer/half_{name}.bin", to3(np.asarray(h.vertices)), to3(np.asarray(h.vertex_normals)),
         np.full((len(h.vertices), 3), 255), h.faces)
import os
for f in sorted(os.listdir("viewer")): print(f, os.path.getsize("viewer/" + f))
