#!/bin/bash
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --mem=2G                                        # Ask for 10 GB of RAM
#SBATCH --time=12:00:00                                   # The job will run for 12 hours

module load python/3.10

source ~/venvs/torchgfn/bin/activate

export PYTHONPATH=$PYTHONPATH:/$HOME/torchgfn/src

python log.py --experiment_name replay_and_capacity --folder logs

