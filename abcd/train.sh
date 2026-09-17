#!/bin/bash

for cfg in single/*; do nohup python3 main.py --config $cfg --flavor single > $(basename $cfg .yaml).log & done
