"""
preprocess_ch3no2.py
Convert raw RCM pressure and volume histories to CSV files
with Time(msec), Temperature(K), Pressure(bar) columns.

For each volume history:
  1. Parse pressure level (15 or 30 bar) and IDT from filename
  2. Match to corresponding pressure trace in Excel by IDT label
  3. Look up T_ref from IgnitionDelay_database by WeakIDT
  4. Compute T(t) using adiabatic core assumption:
     T(t) = T_ref * (P(t) / P_ref) ^ ((gamma-1) / gamma)
     with volume correction from volume history
  5. Output CSV to LW_v1/data/
"""
import os, re, pandas as pd, numpy as np

# Paths
DATA_BASE = r"E:\codex\爆炸\ch3no2_Pressure and volume histories\Pressure and volume histories"
DB_PATH = r"E:\codex\爆炸\IgnitionDelay_database.xlsx"
OUT_DIR = r"E:\codex\爆炸\LW_v1\data"
GAMMA = 1.35  # specific heat ratio

os.makedirs(OUT_DIR, exist_ok=True)

# Load ignition delay database
db = pd.read_excel(DB_PATH)
db = db.dropna(subset=["fuels"])
print(f"Database loaded: {len(db)} entries")

# Build lookup: (phi, Pressure_rounded, WeakIDT_rounded) -> Temperature
db_lookup = {}
for _, row in db.iterrows():
    phi = float(row["phi"])
    pres = float(row["Pressure/bar"])
    widt = float(row["WeakIDT/ms"])
    temp = float(row["Temperature"])
    key = (phi, round(pres), round(widt, 1))
    db_lookup[key] = temp
    # Also add with 0.5 tolerance
    db_lookup[(phi, round(pres), round(widt + 0.5, 1))] = temp
    db_lookup[(phi, round(pres), round(widt - 0.5, 1))] = temp

print(f"Database lookup built: {len(db_lookup)} keys")


def parse_phi_from_cond(cond_name):
    """Extract phi value from condition folder name like 'phi=1.0,2pct CH3NO2'"""
    # Try phi=X.X pattern
    m = re.search(r'phi\s*=\s*([\d.]+)', cond_name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: look for phi character
    m = re.search(r'=([\d.]+)', cond_name)
    if m:
        return float(m.group(1))
    return None


def find_db_temperature(phi, bar, idt_val):
    """Look up temperature from database by matching phi, pressure, and IDT"""
    # Try exact match first
    for (p, pr, widt), temp in db_lookup.items():
        if abs(p - phi) < 0.02 and pr == bar and abs(widt - idt_val) < 1.5:
            return temp
    # Try looser match
    for (p, pr, widt), temp in db_lookup.items():
        if abs(p - phi) < 0.02 and pr == bar and abs(widt - idt_val) < 3.0:
            return temp
    return None


total_generated = 0
all_entries = []

# Iterate over all condition folders
for cond_name in sorted(os.listdir(DATA_BASE)):
    cond_dir = os.path.join(DATA_BASE, cond_name)
    if not os.path.isdir(cond_dir):
        continue

    phi = parse_phi_from_cond(cond_name)
    if phi is None:
        print(f"SKIP: Cannot parse phi from: {cond_name}")
        continue

    print(f"\n{'='*60}")
    print(f"Condition: {cond_name}  (phi={phi})")

    vol_dir = os.path.join(cond_dir, "volume histories")
    pres_dir = os.path.join(cond_dir, "pressure histories")
    if not os.path.isdir(vol_dir) or not os.path.isdir(pres_dir):
        continue

    # Build pressure trace index from Excel files
    # Maps IDT value -> dict of time, pressure arrays
    pres_index = {}
    for pf in os.listdir(pres_dir):
        if not pf.endswith(".xlsx"):
            continue
        xl_path = os.path.join(pres_dir, pf)
        bar_level = 15 if "15bar" in pf else 30
        xl = pd.ExcelFile(xl_path)
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl_path, sheet_name=sheet)
            cols = list(df.columns)
            if len(cols) == 0:
                continue
            # Columns come in pairs: (Time, IDT=XX ms, Time.1, IDT=YY ms, ...)
            i = 0
            while i < len(cols) - 1:
                tc = str(cols[i])
                pc = str(cols[i + 1])
                if "Time" in tc and "IDT" in pc:
                    # Extract IDT value from column name
                    idt_str = pc.replace("IDT=", "").replace(" ms", "").replace("ms", "").strip()
                    try:
                        idt_val_col = float(idt_str)
                        t_arr = df[cols[i]].dropna().values
                        p_arr = df[cols[i + 1]].dropna().values
                        # Store with bar level
                        key = (bar_level, round(idt_val_col, 2))
                        pres_index[key] = {
                            "time": t_arr,
                            "pressure": p_arr,
                            "bar": bar_level
                        }
                    except ValueError:
                        pass
                i += 2

    print(f"  Pressure traces indexed: {len(pres_index)}")

    # Process each volume history file
    for vf in sorted(os.listdir(vol_dir)):
        if not vf.endswith(".csv"):
            continue

        vol_path = os.path.join(vol_dir, vf)
        vol_df = pd.read_csv(vol_path, header=None)
        vol_time = vol_df.iloc[:, 0].values.astype(float)
        volume = vol_df.iloc[:, 1].values.astype(float)

        # Parse IDT from filename
        m = re.search(r'IDT=([\d.]+)\s*ms', vf)
        if not m:
            m = re.search(r'IDT=([\d.]+)', vf)
        if not m:
            print(f"  SKIP: Cannot parse IDT from {vf}")
            continue
        idt_val = float(m.group(1))

        # Parse pressure level from filename
        bar = 15 if "15bar" in vf else 30

        # Find matching pressure trace
        pres_key = (bar, round(idt_val, 2))
        pres_info = pres_index.get(pres_key)
        if pres_info is None:
            # Try fuzzy match
            for (b, k_idt), info in pres_index.items():
                if b == bar and abs(k_idt - idt_val) < 2.0:
                    pres_info = info
                    break

        if pres_info is None:
            print(f"  SKIP: No pressure trace for bar={bar}, IDT={idt_val}ms")
            continue

        # Look up temperature from database
        temp_k = find_db_temperature(phi, bar, idt_val)
        if temp_k is None:
            print(f"  WARN: No DB temperature for phi={phi}, bar={bar}, IDT~{idt_val:.1f}ms - using fallback")
            temp_k = 800.0

        # ---- Align and compute temperature ----
        p_time_raw = pres_info["time"]
        p_val_raw = pres_info["pressure"]

        # Find end-of-compression: global peak pressure in the trace
        # In RCM, TDC is at the global maximum (compression peak)
        p_peak_idx = int(np.argmax(p_val_raw))
        p_ref = p_val_raw[p_peak_idx]
        tdc_time = p_time_raw[p_peak_idx]

        # Trim to post-compression region
        post_mask = p_time_raw >= tdc_time
        p_post_time = p_time_raw[post_mask] - tdc_time
        p_post = p_val_raw[post_mask]

        if len(p_post_time) < 2:
            print(f"  SKIP: Not enough post-compression data")
            continue

        # Interpolate pressure onto volume time grid
        p_interp = np.interp(vol_time, p_post_time, p_post,
                             left=p_post[0], right=p_post[-1])

        # Compute temperature using adiabatic core assumption
        # T(t) = T_ref * (P(t) / P_ref)^((gamma-1)/gamma) * (V(t)/V_ref)^(gamma-1)
        v_ref = volume[0]
        T_ref = temp_k
        temp = T_ref * (p_interp / p_ref) ** ((GAMMA - 1.0) / GAMMA)
        temp = temp * (volume / v_ref) ** (GAMMA - 1.0)

        # Build output filename
        out_name = f"CH3NO2_phi{phi}_bar{bar}_T{int(temp_k)}K_IDT{idt_val:.1f}ms.csv"
        out_path = os.path.join(OUT_DIR, out_name)

        out_df = pd.DataFrame({
            "Time(msec)": vol_time,
            " Temperature(K)": temp,
            " Pressure(bar)": p_interp
        })
        out_df.to_csv(out_path, index=False)

        # Extract Dp info
        # Dp = pressure rise at ignition - find max pressure near IDT
        idx_idt = np.argmin(np.abs(vol_time - idt_val))
        p_at_idt = p_interp[idx_idt]
        dp_val = p_at_idt - p_ref

        print(f"  OK: {out_name}")
        print(f"      T_ref={T_ref:.1f}K  P_ref={p_ref:.2f}bar  IDT={idt_val:.1f}ms  Dp={dp_val:.2f}bar")

        total_generated += 1
        all_entries.append({
            "filename": out_name,
            "phi": phi,
            "bar": bar,
            "T_ref": T_ref,
            "P_ref": p_ref,
            "idt": idt_val,
            "dp": dp_val
        })

# Print summary
print(f"\n{'='*60}")
print(f"GENERATION COMPLETE: {total_generated} CSV files")
print(f"{'='*60}")

# Group entries by phi and bar for JSON config
for phi_val in sorted(set(e["phi"] for e in all_entries)):
    for bar_val in [15, 30]:
        entries = [e for e in all_entries if e["phi"] == phi_val and e["bar"] == bar_val]
        if entries:
            print(f"\nphi={phi_val}, bar={bar_val}:")
            for e in sorted(entries, key=lambda x: x["idt"]):
                print(f"  T={e['T_ref']:.0f}K  IDT={e['idt']:.1f}ms  P_ref={e['P_ref']:.2f}bar  Dp={e['dp']:.2f}bar")
