#!/bin/bash
curdir=$(pwd)
echo currently in $curdir
curtime=$(date "+%y-%m-%d_%H-%M-%S")

rangeA=(0.1)
rangeE=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
rangeI=(0.2 0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0)
rangeP=(2 4 6 8 10)

for valA in "${rangeA[@]}"; do
	for valE in "${rangeE[@]}"; do
		for valI in "${rangeI[@]}"; do
			for valP in "${rangeP[@]}"; do
				#if (( $(echo "$valI >= $valE" |bc -l) )); then

					# check not done before in a sweep
					if  false; then #echo "${oldrangeA[@]}" | grep -q "$valA" && 
				    	# echo "${oldrangeE[@]}" | grep -q "$valE" && 
					    # echo "${oldrangeI[@]}" | grep -q "$valI" && 
						# echo "${oldrangeP[@]}" | grep -q "$valP"; then
					    : #echo $valA $valE $valI $valP "done before"
					else
						echo exporting settings $valA $valE $valI $valP
						python3 export_sim_settings.py --mA $valA --mE $valE --mI $valI --mP $valP
						echo submitting job:
						savetodir=$(cat cur_saveto_dir.txt)
						rm cur_saveto_dir.txt
						cd $savetodir; sbatch submit_job.sh; cd $curdir
					fi
				#fi
			done
		done
	done
done