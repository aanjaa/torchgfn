#!/bin/bash
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                   # The job will run for 24 hours

module load python/3.10

source ~/venvs/torchgfn/bin/activate

export PYTHONPATH=$PYTHONPATH:/$HOME/torchgfn/src

python main.py --local_debug false --use_wandb true --experiment_name searchspaces_losses
#python main.py --local_debug false --use_wandb true --experiment_name replay_and_capacity
#python main.py --local_debug false --use_wandb true --experiment_name exploration_strategies
