#!/bin/bash

export PYTHONPATH="~/PycharmProjects/concept-lens:$PYTHONPATH"
#export PYTHONPATH="~/PycharmProjects/concept-lens/data_generation/generator/torch_utils/:$PYTHONPATH"
#export PYTHONPATH="~/PycharmProjects/concept-lens/data_generation/generator/dnnlib/:$PYTHONPATH"

#python data_generation/experiment_walk_distance.py --domain s3t_wild512 --layer early --seed 0

#python data_generation/vector_arithmetic.py --domain s3t_wild512 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3t_ffhq1024 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3t_metfaces1024 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3r_wild512 --layer early --seed 0

python data_generation/supervised.py --domain s2_wild512 --layer early --seed 0 --edit_dist 10

python data_generation/vector_arithmetic.py --domain s2_ffhq1024 --layer early --seed 0 --edit_dist 5
python data_generation/vector_arithmetic.py --domain s2_metfaces1024 --layer early --seed 0
python data_generation/vector_arithmetic.py --domain s2_wild512 --layer early --seed 0
