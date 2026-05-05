import h5py
import numpy as np

def check_calibration(real_path, syn_path):
    with h5py.File(real_path, 'r') as fr, h5py.File(syn_path, 'r') as fs:
        ry = fr['labels'][:]
        rt = fr['tokens'][:]
        
        sy = fs['labels'][:]
        st = fs['tokens'][:]
        
        print(f"Checking calibration: {syn_path}")
        for cid in range(6):
            rm = ry == cid
            sm = sy == cid
            
            if rm.sum() > 0 and sm.sum() > 0:
                r_mean = rt[rm].mean()
                s_mean = st[sm].mean()
                r_std = rt[rm].std()
                s_std = st[sm].std()
                
                diff_mean = abs(r_mean - s_mean)
                diff_std = abs(r_std - s_std)
                
                status = "CALIBRATED" if diff_mean < 1e-3 and diff_std < 1e-3 else "NOT CALIBRATED"
                print(f"  Class {cid}: Mean Diff={diff_mean:.6f}, Std Diff={diff_std:.6f} -> {status}")

check_calibration('results_wisdm_v2_6class/token_store.hdf5', 'synthetic_tokens_v2_6class_5pct/synthetic_tokens.hdf5')
check_calibration('results_wisdm_v2_6class/token_store.hdf5', 'synthetic_tokens_v2_6class_calibrated/synthetic_tokens.hdf5')
