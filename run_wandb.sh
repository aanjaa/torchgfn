#!/bin/bash
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                                   # The job will run for 24 hours

module load python/3.10

source ~/venvs/torchgfn/bin/activate

export PYTHONPATH=$PYTHONPATH:/$HOME/torchgfn/src

experiment_names=("smoothness_losses" "reward_losses")

# Loop over the elements in the array
for experiment_name in "${experiment_names[@]}"; do
  echo "$experiment_name"
  python main.py --experiment_name "$experiment_name" --local_debug false --use_wandb true
done

