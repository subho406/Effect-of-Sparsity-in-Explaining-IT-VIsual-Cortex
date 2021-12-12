#!/bin/bash
#SBATCH --account=def-afyshe-ab
#SBATCH --cpus-per-task=48   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128000M        # memory per node
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID


module load python/3.6
module load cuda cudnn 
source ~/base/bin/activate
python /home/subho/projects/def-afyshe-ab/subho/CMPUT652/Effect-of-Sparsity-in-Explaining-IT-VIsual-Cortex/calculate_rsa_results.py