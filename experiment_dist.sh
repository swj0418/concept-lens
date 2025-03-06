#!/bin/bash

export PYTHONPATH="~/PycharmProjects/concept-lens:$PYTHONPATH"
#export PYTHONPATH="~/PycharmProjects/concept-lens/data_generation/generator/torch_utils/:$PYTHONPATH"
#export PYTHONPATH="~/PycharmProjects/concept-lens/data_generation/generator/dnnlib/:$PYTHONPATH"

python data_generation/experiment_walk_distance.py --domain s3t_wild512 --layer early --seed 0
