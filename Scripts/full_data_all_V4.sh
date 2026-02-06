#!/bin/bash
# Run all V4 datasets through data_pipe_V2
# Uses fixed mcap files (folder_name/folder_name.mcap)
# Memory allocated at ~20GB per minute of episode length

V4DIR=/arm/u/weizhuo2/Documents/Data_pipe/Bags/V4Data
JOB=full_data_1_job_V2.sh

# [GOOD]                                                          duration  mem
sbatch --mem=80G  $JOB "$V4DIR/V4Data_260203srcbay/V4Data_260203srcbay.mcap"          # 213s  ~4min
sbatch --mem=120G $JOB "$V4DIR/V4Data_260203packardbase/V4Data_260203packardbase.mcap" # 356s  ~6min
sbatch --mem=200G $JOB "$V4DIR/V4Data_260203night570/V4Data_260203night570.mcap"       # 581s ~10min
sbatch --mem=100G $JOB "$V4DIR/V4Data_260203codabase2/V4Data_260203codabase2.mcap"     # 257s  ~4min
sbatch --mem=100G $JOB "$V4DIR/V4Data_260203coda4/V4Data_260203coda4.mcap"             # 273s  ~5min
sbatch --mem=220G $JOB "$V4DIR/V4Data_260205quad2coda/V4Data_260205quad2coda.mcap"     # 628s ~10min
sbatch --mem=80G  $JOB "$V4DIR/V4Data_260205gatesbase/V4Data_260205gatesbase.mcap"     # 110s  ~2min
sbatch --mem=180G $JOB "$V4DIR/V4Data_260205quad/V4Data_260205quad.mcap"               # 543s  ~9min
sbatch --mem=120G $JOB "$V4DIR/V4Data_260205greenlib/V4Data_260205greenlib.mcap"       # 351s  ~6min
sbatch --mem=180G $JOB "$V4DIR/V4Data_260205tressider/V4Data_260205tressider.mcap"     # 531s  ~9min
sbatch --mem=80G  $JOB "$V4DIR/V4Data_260205gates2nd/V4Data_260205gates2nd.mcap"       # 244s  ~4min
sbatch --mem=120G $JOB "$V4DIR/V4Data_260205gates1st/V4Data_260205gates1st.mcap"       # 360s  ~6min

# [maybe good] - minor stuttering
sbatch --mem=220G $JOB "$V4DIR/V4Data_260203packard3/V4Data_260203packard3.mcap"       # 614s ~10min
sbatch --mem=160G $JOB "$V4DIR/V4Data_260203coda01/V4Data_260203coda01.mcap"           # 481s  ~8min
sbatch --mem=80G  $JOB "$V4DIR/V4Data_260203codabase/V4Data_260203codabase.mcap"       # 161s  ~3min

# stuttering / camera pitch - uncomment if needed
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_002417redo/V4Data_20260204_002417redo.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_260203nightcoda/V4Data_260203nightcoda.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_161816redo/V4Data_20260204_161816redo.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_163332redo/V4Data_20260204_163332redo.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_164150redo/V4Data_20260204_164150redo.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_165736redo/V4Data_20260204_165736redo.mcap"
# sbatch --mem=200G $JOB "$V4DIR/V4Data_20260204_172158redo/V4Data_20260204_172158redo.mcap"
