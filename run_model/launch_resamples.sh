#!/bin/bash
curdir=$(pwd)
echo currently in $curdir
curtime=$(date "+%y-%m-%d_%H-%M-%S")

adjustPNinput=1
noclass=250
nORNonly=200
nLNonly=200
nLNsparse=0
nLNbroad=0
nLNpatchy=0
nLNregional=0
nPNonly=200
nORNminus=0
nLNminus=0
nPNminus=0
nall=200
nPNuPNonly=0
nPNmPNonly=0

for i in $(seq 1 $noclass); do 
	echo running blank resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 0 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nORNonly); do 
	echo running ORN resample...
	python3 export_resampling_sim_settings.py --rO 1 --rL 0 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nLNonly); do 
	echo running LN resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nLNsparse); do 
	echo running LN sparse resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rLsparse 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done
for i in $(seq 1 $nLNbroad); do 
	echo running LN broad resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rLbroad 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done
for i in $(seq 1 $nLNpatchy); do 
	echo running LN patchy resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rLpatchy 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done
for i in $(seq 1 $nLNregional); do 
	echo running LN regional resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rLregional 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNonly); do 
	echo running PN resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 0 --rP 1 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNuPNonly); do 
	echo running uPN only resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 0 --rP 1 --ruP 1 --rmP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNmPNonly); do 
	echo running mPN only resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 0 --rP 1 --ruP 0 --rmP 1 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nORNminus); do 
	echo running ORN minus resample...
	python3 export_resampling_sim_settings.py --rO 0 --rL 1 --rP 1 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nLNminus); do 
	echo running LN minus resample...
	python3 export_resampling_sim_settings.py --rO 1 --rL 0 --rP 1 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nPNminus); do 
	echo running PN minus resample...
	python3 export_resampling_sim_settings.py --rO 1 --rL 1 --rP 0 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done

for i in $(seq 1 $nall); do 
	echo running ALL resample...
	python3 export_resampling_sim_settings.py --rO 1 --rL 1 --rP 1 --adjustPNInputs $adjustPNinput
	echo submitting job:
	savetodir=$(cat cur_saveto_dir.txt)
	rm cur_saveto_dir.txt
	cd $savetodir
	sbatch submit_job.sh
	cd $curdir
done