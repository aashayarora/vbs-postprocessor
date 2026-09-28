#!/bin/bash
#
# Train every single/ config with run_abcd.py in the pixi env (pixi.toml), in the
# background, logging to <config>.log. Each config picks its GPU with its `devices` key.
#
# prp-gpu-1 is shared (125 GB): the 1lep_1FJ and Run 3 0lep trainings take ~20-30 GB of host
# memory each while loading. `MIN_FREE_GB=30 nohup ./train.sh &` waits for that much free
# memory before each start, and 5 minutes after it, so the loading peaks do not stack up.

cd "$(dirname "${BASH_SOURCE[0]}")"
MIN_FREE_GB=${MIN_FREE_GB:-0}
for cfg in single/*; do
    if [ "$MIN_FREE_GB" -gt 0 ]; then
        until [ "$(free -g | awk '/Mem:/{print $7}')" -ge "$MIN_FREE_GB" ]; do sleep 60; done
        echo "$(date +%H:%M) starting $cfg"
    fi
    setsid nohup pixi run python run_abcd.py --config $cfg --flavor single \
        > $(basename $cfg .yaml).log 2>&1 < /dev/null &
    [ "$MIN_FREE_GB" -gt 0 ] && sleep 300
done
