#!/bin/bash

echo "Installing Genesis in genesis_test environment..."

# Activate the genesis_test environment and install Genesis
conda run -n genesis_test pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git

echo "Genesis installation complete!"
echo "To use the environment: conda activate genesis"
