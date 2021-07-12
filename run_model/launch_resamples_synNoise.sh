#!/bin/bash
curdir=$(pwd)
echo currently in $curdir
curtime=$(date "+%y-%m-%d_%H-%M-%S")

synStrength=1
noclass=80
nORNonly=0
nLNonly=0
nPNonly=0
nORNminus=0
nLNminus=0
nPNminus=0
nall=130

for i in $(seq 1 $noclass); do 
	echo running blank synaptic noise...
	python3 export_resampling_sim_settings.py --sO 0 --sL 0 --sP 0 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nORNonly); do 
	echo running ORN synaptic noise...
	python3 export_resampling_sim_settings.py --sO 1 --sL 0 --sP 0 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nLNonly); do 
	echo running LN synaptic noise...
	python3 export_resampling_sim_settings.py --sO 0 --sL 1 --sP 0 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNonly); do 
	echo running PN synaptic noise...
	python3 export_resampling_sim_settings.py --sO 0 --sL 0 --sP 1 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nORNminus); do 
	echo running ORN minus synaptic noise...
	python3 export_resampling_sim_settings.py --sO 0 --sL 1 --sP 1 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nLNminus); do 
	echo running LN minus synaptic noise...
	python3 export_resampling_sim_settings.py --sO 1 --sL 0 --sP 1 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNminus); do 
	echo running PN minus synaptic noise...
	python3 export_resampling_sim_settings.py --sO 1 --sL 1 --sP 0 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nall); do 
	echo running ALL synaptic noise...
	python3 export_resampling_sim_settings.py --sO 1 --sL 1 --sP 1 --sStrength $synStrength
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done