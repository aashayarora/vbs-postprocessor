#!/bin/bash
#
# Train every single/ config at once, in the pixi env (pixi.toml). Each config picks its own
# GPU with its `devices` key. On prp-gpu-1 (125 GB, shared) the 1lep_1FJ and Run 3 0lep
# trainings need ~20-30 GB each, so check free memory before starting them all together.

cd "$(dirname "${BASH_SOURCE[0]}")"
for cfg in single/*; do
    setsid nohup pixi run python main.py --config $cfg --flavor single \
        > $(basename $cfg .yaml).log 2>&1 < /dev/null &
done
