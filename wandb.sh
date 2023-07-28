export PYTHONPATH=$PYTHONPATH:/home/anja/Documents/GFlowNets/torchgfn/src

experiment_name="searchspaces_losses" #["reward_losses", "searchspaces_losses", "smoothness_losses"]

experiment_dir=$(realpath "$(pwd)/logs/$experiment_name")
for name in $(ls "$experiment_dir"); do
  echo "$name"
  python main.py --experiment_name "$experiment_name" --name "$name" --local_debug false --use_wandb true
done