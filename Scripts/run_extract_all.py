"""
Batch extract video frames and DINOv3 features for all datasets in segRedo_full_fpath_lst.

Usage:
    python run_extract_all.py
"""

import subprocess
import sys

DATASETS = [
    "V2DataRedo_field_lag",
    "V2DataRedo_realsense0712_lag",
    "V2DataRedo_human0408_lag",
    "V2DataRedo_human0401_lag",
    "V2DataRedo_human0306final_lag",
    "V2DataNew_231226psyc1C_lag",
    "V2DataNew_231226psyc2C_lag",
    "V2DataNew_231228bachtelC_lag",
    "V2DataNew_231228lawnewC_lag",
    "V2DataNew_231228lawold_needpanofixC_lag",
    "V2DataNew_231228lawstairC_lag",
    "V2DataNew_231228quadC_lag",
    "V2DataNew_231230frat1C_lag",
    "V2DataNew_231230frat2C_lag",
    "V2DataNew_231231ccrma1C_lag",
    "V2DataNew_240103flomoC_lag",
    "V2DataNew_240104darkC_lag",
    "V2DataNew_240104dormC_lag",
    "V2DataNew_240104hospital1C_lag",
    "V2DataNew_240104hospital2C_lag",
    "V2DataNew_240104lomita_needpanofixC_lag",
    "V2DataNew_240105clarkC_lag",
    "V2DataNew_240105clark_shortC_lag",
    "V2DataNew_240105hospitalday1C_lag",
    "V2DataNew_240105hospitalday2C_lag",
]

if __name__ == "__main__":
    for i, dataset in enumerate(DATASETS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(DATASETS)}] Processing: {dataset}")
        print(f"{'='*60}")
        subprocess.run([sys.executable, "extract_video_mmap.py", dataset])
