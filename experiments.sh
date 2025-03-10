#!/bin/bash

export PYTHONPATH="~/PycharmProjects/concept-lens:$PYTHONPATH"

SEED=121

#python data_generation/vector_arithmetic.py --domain s2_ffhq256 --layer early --seed 0
python data_generation/sefa.py              --domain s2_ffhq256 --layer early --seed ${SEED}
#python data_generation/ganspace.py          --domain s2_ffhq256 --layer early --seed 0

python data_generation/vector_arithmetic.py --domain s2_ffhq256 --layer middle --seed ${SEED}
#python data_generation/sefa.py              --domain s2_ffhq256 --layer middle --seed 0
#python data_generation/ganspace.py          --domain s2_ffhq256 --layer middle --seed 0

#python data_generation/vector_arithmetic.py --domain s3t_ffhq1024 --layer early --seed 0
#python data_generation/sefa.py --domain s3t_ffhq1024 --layer early --seed 0
#python data_generation/ganspace.py --domain s3t_ffhq1024 --layer early --seed 0
#
#python data_generation/vector_arithmetic.py --domain s3t_ffhq1024 --layer middle --seed 0
#python data_generation/sefa.py --domain s3t_ffhq1024 --layer middle --seed 0
#python data_generation/ganspace.py --domain s3t_ffhq1024 --layer middle --seed 0

#python data_generation/vector_arithmetic.py --domain s2_wild512 --layer early --seed 0
python data_generation/sefa.py              --domain s2_wild512 --layer early --seed ${SEED}
#python data_generation/ganspace.py          --domain s2_wild512 --layer early --seed 0

python data_generation/vector_arithmetic.py --domain s2_wild512 --layer middle --seed ${SEED}
#python data_generation/sefa.py              --domain s2_wild512 --layer middle --seed 0
#python data_generation/ganspace.py          --domain s2_wild512 --layer middle --seed 0

#python data_generation/vector_arithmetic.py --domain s2_cat512 --layer early --seed 0
#python data_generation/sefa.py              --domain s2_cat512 --layer early --seed 0
#python data_generation/ganspace.py          --domain s2_cat512 --layer early --seed 0
#
#python data_generation/vector_arithmetic.py --domain s2_cat512 --layer middle --seed 0
#python data_generation/sefa.py              --domain s2_cat512 --layer middle --seed 0
#python data_generation/ganspace.py          --domain s2_cat512 --layer middle --seed 0

#python data_generation/vector_arithmetic.py --domain s3t_wild512 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3t_ffhq1024 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3t_metfaces1024 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s3r_wild512 --layer early --seed 0
#
#python data_generation/vector_arithmetic.py --domain s2_ffhq1024 --layer early --seed 0 --edit_dist 5
#python data_generation/vector_arithmetic.py --domain s2_metfaces1024 --layer early --seed 0
#python data_generation/vector_arithmetic.py --domain s2_wild512 --layer early --seed 0
