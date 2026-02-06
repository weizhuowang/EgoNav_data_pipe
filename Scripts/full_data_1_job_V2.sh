#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm2

#SBATCH --job-name=datapipe_v2
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=2:00:00

#SBATCH --ntasks=1         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=7  # max number of cores each task below can use
#SBATCH --mem=200G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task >= total cores used below +3.

#SBATCH --gres=gpu:1       # Request 1 GPU for the job

# Usage: sbatch full_data_1_job_V2.sh <bag_filename> [--no-seg] [--prefix PREFIX]
# Example:
#   sbatch full_data_1_job_V2.sh V2DataRedo_field.bag
#   sbatch full_data_1_job_V2.sh V4Data_260205quad.mcap --prefix V4TEST
#   sbatch full_data_1_job_V2.sh V2DataRedo_field.bag --no-seg

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <bag_filename> [extra args for data_pipe_V2.py]"
    exit 1
fi

source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate GSAM
which python

bag_name=$1
shift  # remaining args passed to data_pipe_V2.py

BAG_DIR=/arm/u/weizhuo2/Documents/Data_pipe/Bags
OUT_DIR=/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/V2_test
SCRIPT_DIR=/arm/u/weizhuo2/Documents/Data_pipe/Scripts/V2

# Support both absolute path and filename relative to BAG_DIR
if [[ "$bag_name" = /* ]]; then
    fpath="$bag_name"
else
    fpath="$BAG_DIR/$bag_name"
fi
echo "Processing: $fpath"
echo "Output dir: $OUT_DIR"

python "$SCRIPT_DIR/data_pipe_V2.py" \
    --bag "$fpath" \
    --output "$OUT_DIR" \
    "$@"

echo "Done"
