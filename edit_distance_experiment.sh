#!/bin/bash

export PYTHONPATH="$HOME/PycharmProjects/concept-lens:$PYTHONPATH"

SEED=4180

python data_generation/sefa.py              --domain s2_ffhq256 --layer early --seed ${SEED}
python data_generation/vector_arithmetic.py --domain s2_ffhq256 --layer middle --seed ${SEED}
python data_generation/sefa.py              --domain s2_wild512 --layer early --seed ${SEED}
python data_generation/vector_arithmetic.py --domain s2_wild512 --layer middle --seed ${SEED}
