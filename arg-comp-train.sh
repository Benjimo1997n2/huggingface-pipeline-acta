#!/usr/bin/env bash

set -ex

# Activate the virtual environment if necessary
module load conda/5.0.1-python3.6 
module load cuda/10.2
source activate huggingface-pipeline-acta

# Run the Python script
python ./arg-comp-train.py

# Deactivate the virtual environment if necessary
# deactivate
