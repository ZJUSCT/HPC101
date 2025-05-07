#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=Solver

# Run BICGSTAB
./build/bicgstab $1