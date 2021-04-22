#!/bin/bash
echo exporting settings:
python3 export_sim_settings.py
echo submitting job:
savetodir=$(cat cur_saveto_dir.txt)
rm cur_saveto_dir.txt
cd $savetodir
sbatch submit_job.sh