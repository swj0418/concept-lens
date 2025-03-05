#!/bin/bash

export PYTHONPATH="~/PycharmProjects/concept-lens:$PYTHONPATH"
python data_generation/generator/experiment_walk_distance.py --domain s3t_wild512 --layer early --seed 0
