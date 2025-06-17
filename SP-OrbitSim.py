import math, numpy as np, pandas as pd, os

# ------------- USER CONFIG -------------
ANTENNAS = ['+X']   # define faces here (any of +X, -X, +Y, -Y, +Z, -Z)
HALF_CONE_DEG = 80.0      # antenna half‑cone
ALT_SC = 500.0            # km altitude
INC_DEG = 56.0            # inclination
TIME_STEP = 60            # seconds per step
# ---------------------------------------

MU = 398600.4418  # km^3/s^2
RE = 6378.0       # km

def rot1(a): return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
def rot3(a): return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
def eci_pos(a,inc,raan,argp,nu):
    r_perif = np.array([a*math.cos(nu), a*math.sin(nu), 0.0])
    return rot3(raan) @ rot1(inc) @ rot3(argp) @ r_perif

def build_cons(tag,h_km,inc_deg,raans_deg,sats_per_plane):
    a = RE + h_km
    inc = math.radians(inc_deg)
    out=[]
    for p, raan_deg in enumerate(raans_deg):
        raan = math.radians(raan_deg)
        for k in range(sats_per_plane):
            nu = 2*math.pi*k/sats_per_plane
            label=f'{tag}-P{p+1}{k+1}'
            out.append((a,inc,raan,0.0,nu,label))
    return out

GAL = build_cons('GAL',23222,56,[0,120,240],9)
GPS = build_cons('GPS',20200,55,[i*60 for i in range(6)],8)
BDS_M= build_cons('BDS-M',21528,55,[0,120,240],7)
BDS_I= build_cons('BDS-I',35786,55,[60,180,300],2)
SATLIST = GAL+GPS+BDS_M+BDS_I

A_SC = RE + ALT_SC
INC_SC = math.radians(INC_DEG)
RAAN_SC = 0.0
ARGP_SC = 0.0
NU0_SC = 0.0
n_sc = math.sqrt(MU/A_SC**3)
ORBIT_PERIOD = 2*math.pi/n_sc
HALFCONE = math.radians(HALF_CONE_DEG)

def face_vec(face, x_b, y_b, z_b):
    mapper = { '+X': x_b, '-X': -x_b,
               '+Y': y_b, '-Y': -y_b,
               '+Z': z_b, '-Z': -z_b }
    return mapper[face]

def visible_at(t_sec):
    nu_sc = (NU0_SC + n_sc*t_sec) % (2*math.pi)
    r_sc = eci_pos(A_SC, INC_SC, RAAN_SC, ARGP_SC, nu_sc)
    v_sc = np.cross([0,0,1], r_sc); v_sc/=np.linalg.norm(v_sc)
    r_hat = r_sc/np.linalg.norm(r_sc)
    z_b = -r_hat; x_b = v_sc; y_b = np.cross(z_b,x_b); y_b/=np.linalg.norm(y_b)
    bores = {f: face_vec(f,x_b,y_b,z_b) for f in ANTENNAS}
    vis = {f: [] for f in ANTENNAS}
    for a,inc,raan,argp,nu,label in SATLIST:
        r_sat = eci_pos(a,inc,raan,argp,nu)
        los = r_sat - r_sc
        if np.dot(los,r_hat) <= 0: continue
        los_hat = los/np.linalg.norm(los)
        for f,b in bores.items():
            if math.acos(np.clip(np.dot(los_hat,b),-1,1)) <= HALFCONE:
                vis[f].append(label)
    return vis

steps = int(round(ORBIT_PERIOD / TIME_STEP)) + 1
rows=[]
for k in range(steps):
    t=k*TIME_STEP
    vis = visible_at(t)
    row = {'Minute':k}
    union=set()
    for f in ANTENNAS:
        labs=sorted(vis[f])
        row[f'Count_{f}'] = len(labs)
        row[f'Labels_{f}'] = ';'.join(labs)
        union.update(labs)
    row['Count_Union']=len(union)
    row['Labels_Union']=';'.join(sorted(union))
    rows.append(row)

df=pd.DataFrame(rows)
print('Faces used:', ANTENNAS)
print(df[['Minute','Count_Union']+[f'Count_{f}' for f in ANTENNAS]])
