#!/bin/bash

#SBATCH -J vqc_training_capstone
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -t 6:00:00

echo "Running python script"
python3 -u vqc_training.py

echo ""
echo "Job ID:              $SLURM_JOB_ID"
echo "Job Name:            $SLURM_JOB_NAME"
echo "Number of Nodes:     $SLURM_JOB_NUM_NODES"
echo "Number of CPU cores: $SLURM_CPUS_ON_NODE"
echo "Number of Tasks:     $SLURM_NTASKS"
echo "Partition:           $SLURM_JOB_PARTITION"