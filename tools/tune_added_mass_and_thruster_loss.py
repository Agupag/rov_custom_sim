#!/usr/bin/env python3
import os, re, math, csv
from pathlib import Path

# Find latest log
logs = sorted(Path('.').glob('rov_sim_log_*.txt'))
if not logs:
    print('No log files found in cwd')
    exit(1)
log_file = str(logs[-1])
print('Using log:', log_file)

# Read rov_sim to get thruster layout and thrust limits
rov = Path('rov_sim.py').read_text()
# Extract thrusters block by simple regex
thr_pattern = re.compile(r'THRUSTERS\s*=\s*\[([\s\S]*?)\]\n', re.M)
m = thr_pattern.search(rov)
thrusters = []
if m:
    inner = m.group(1)
    # crude parse: find lines with "pos" and "dir" and "kind"
    item_pattern = re.compile(r"\{([^}]+)\}", re.M)
    for it in item_pattern.finditer(inner):
        txt = it.group(1)
        # pos
        pos_m = re.search(r"'pos'\s*:\s*\(([^)]+)\)|\"pos\"\s*:\s*\(([^)]+)\)", txt)
        dir_m = re.search(r"'dir'\s*:\s*\(([^)]+)\)|\"dir\"\s*:\s*\(([^)]+)\)", txt)
        kind_m = re.search(r"'kind'\s*:\s*'([^']+)'|\"kind\"\s*:\s*\"([^\"]+)\"", txt)
        pos = (0.0,0.0,0.0)
        dirv = (1.0,0.0,0.0)
        kind = 'H'
        if pos_m:
            g = pos_m.group(1) or pos_m.group(2)
            nums = [float(x) for x in g.split(',')]
            pos = tuple(nums)
        if dir_m:
            g = dir_m.group(1) or dir_m.group(2)
            nums = [float(x) for x in g.split(',')]
            dirv = tuple(nums)
        if kind_m:
            kind = kind_m.group(1) or kind_m.group(2)
        thrusters.append({'pos':pos,'dir':dirv,'kind':kind})
else:
    print('Could not parse THRUSTERS from rov_sim.py; using fallback')

# Extract MAX_THRUST_H / V
def extract_float(name):
    m = re.search(rf'{name}\s*=\s*([0-9.]+)', rov)
    return float(m.group(1)) if m else None
MAX_THRUST_H = extract_float('MAX_THRUST_H') or 30.0
MAX_THRUST_V = extract_float('MAX_THRUST_V') or 20.0
MASS = extract_float('MASS') or 12.0

print('Found thrusters:', thrusters)
print('MAX_THRUST_H, V, MASS:', MAX_THRUST_H, MAX_THRUST_V, MASS)

# Parse CSV from log
with open(log_file,'r') as f:
    txt = f.read()
idx = txt.find('# DETAILED_PHYSICS_CSV')
if idx==-1:
    # fallback: find first long CSV-like line
    lines = txt.splitlines()
    csv_lines = [L for L in lines if L.count(',')>15]
else:
    csv_lines = txt[idx:].splitlines()[1:]

v_body = []
thr_levels = []
times = []
for line in csv_lines:
    if not line.strip() or line.startswith('#'):
        continue
    parts = list(csv.reader([line]))[0] if 'csv' in globals() else line.split(',')
    try:
        if len(parts) < 20:
            continue
        t = float(parts[0])
        vx_b = float(parts[8])
        vy_b = float(parts[9])
        vz_b = float(parts[10])
        times.append(t)
        v_body.append((vx_b, vy_b, vz_b))
        # thruster levels last column
        thr = parts[-1].strip()
        if thr.startswith('"') and thr.endswith('"'):
            thr = thr[1:-1]
        vals = [float(x) for x in thr.split(';') if x.strip()]
        thr_levels.append(vals)
    except Exception:
        continue

if not times:
    print('No physics CSV lines parsed')
    exit(1)

# Compute per-axis accel via finite differences
dt = times[1]-times[0] if len(times)>1 else 0.1
ax = []
ay = []
az = []
for i in range(1,len(v_body)):
    ax.append((v_body[i][0]-v_body[i-1][0])/dt)
    ay.append((v_body[i][1]-v_body[i-1][1])/dt)
    az.append((v_body[i][2]-v_body[i-1][2])/dt)

peak_ax = max((abs(x) for x in ax), default=0.0)
peak_ay = max((abs(y) for y in ay), default=0.0)
peak_az = max((abs(z) for z in az), default=0.0)

print('Peak body-frame accels (m/s^2):', peak_ax, peak_ay, peak_az)

# Compute theoretical per-axis max thruster force (body frame)
f_x = f_y = f_z = 0.0
for i,t in enumerate(thrusters):
    kind = t['kind']
    d = t['dir']
    thrust_max = MAX_THRUST_H if kind=='H' else MAX_THRUST_V
    f_x += thrust_max * abs(d[0])
    f_y += thrust_max * abs(d[1])
    f_z += thrust_max * abs(d[2])

print('Estimated max thruster force per-axis (N):', f_x, f_y, f_z)

# Heuristic: want to reduce observed peak accel to ~60% via added mass
target_frac = 0.6
rec_ax = max(0.0, f_x/(max(peak_ax*target_frac, 1e-3)) - MASS)
rec_ay = max(0.0, f_y/(max(peak_ay*target_frac, 1e-3)) - MASS)
rec_az = max(0.0, f_z/(max(peak_az*target_frac, 1e-3)) - MASS)

# Clamp to reasonable bounds
def clamp(x,lo,hi): return lo if x<lo else hi if x>hi else x
rec_ax = clamp(rec_ax, 0.0, MASS*4)
rec_ay = clamp(rec_ay, 0.0, MASS*4)
rec_az = clamp(rec_az, 0.0, MASS*4)

print('\nRecommended ADDED_MASS_BODY (per-axis):', (rec_ax, rec_ay, rec_az))

# Thruster speed loss heuristic: examine mean thruster level during steady state (last 1s)
if thr_levels:
    # compute mean of max abs thruster level per sample
    per_sample = [max(abs(x) for x in row) for row in thr_levels]
    # take last 10% samples as steady
    n = max(1, int(len(per_sample)*0.15))
    steady_mean = sum(per_sample[-n:]) / n
else:
    steady_mean = 0.0

print('Steady mean thruster level (last samples):', steady_mean)
# If thruster level is high (>0.7) at steady state, we may want some speed loss
loss = 0.08 * max(0.0, (steady_mean - 0.6)/0.4)
loss = clamp(loss, 0.0, 0.2)
print('Recommended THRUSTER_SPEED_LOSS_COEF:', round(loss,4))

# Print summary for patching
print('\nPATCH SUGGESTION:')
print(f'ADDED_MASS_BODY = ({rec_ax:.2f}, {rec_ay:.2f}, {rec_az:.2f})')
print(f'THRUSTER_SPEED_LOSS_COEF = {loss:.4f}')
