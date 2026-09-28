#!/bin/bash

# for cfg in single/*; do nohup pixi run python run_abcd.py --config $cfg --flavor single --infer >> $(basename $cfg .yaml).log & done
# for cfg in single/*; do nohup pixi run python run_abcd.py --config $cfg --flavor single --infer --data >> $(basename $cfg .yaml).log & done