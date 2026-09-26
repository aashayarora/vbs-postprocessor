#!/usr/bin/env python3
"""Dask scheduler with HTCondor workers for ``postprocess.py --scheduler``.

Starts the scheduler here, writes its address to --address-file for the drivers, and keeps
--jobs condor worker jobs submitted until it is stopped with SIGINT/SIGTERM, which cancels
them. The jobs are all submitted up front rather than scaled to the queue: the pool only
brings up pilots under real pressure. One cluster serves every driver at once.

The workers run the LCG view below from CVMFS inside an el9 container, so nothing from this
checkout or the pixi env is needed on them. The drivers and this script must run in the same
LCG view (dask pickles code between them).
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
import time

from dask_jobqueue import HTCondorCluster

LCG_VIEW = "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh"
# The glideins reject unpacked.cern.ch/.../cmssw/el9:x86_64 ("Unable to access the Singularity
# image"), although the preselection's el8 counterpart works; the LCG view needs el9.
SINGULARITY_IMAGE = "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel9"
SITES = "T2_US_UCSD"
# Same exclusion as the preselection's condor/submit.py (broken worker node).
EXCLUDED_MACHINES = ["mh-7662-12.t2.ucsd.edu"]
SUBMIT_BATCH = 100


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address-file", required=True, help="write the scheduler address here")
    p.add_argument("--jobs", type=int, default=100, help="condor worker jobs to submit (default: 100)")
    p.add_argument("--cores", type=int, default=4,
                   help="cores per job; each runs one single-threaded worker process (default: 4)")
    p.add_argument("--memory", default="12GB",
                   help="memory per job, split over its workers. The largest input (1.4 GB) peaks "
                        "at 1.8 GB per worker; two such jobs fit a 10-core / 25 GB UCSD pilot "
                        "(default: 12GB)")
    p.add_argument("--disk", default="4GB", help="disk per job (default: 4GB)")
    p.add_argument("--lifetime", default="1h",
                   help="restart each worker process after this long, within its job, bounding "
                        "cling's JIT growth (default: 1h)")
    p.add_argument("--log-dir", default="dask-logs", help="condor job logs (default: dask-logs)")
    p.add_argument("--dashboard-port", type=int, default=8787, help="scheduler dashboard port (default: 8787)")
    args = p.parse_args(argv)

    os.makedirs(args.log_dir, exist_ok=True)
    requirements = "(HAS_SINGULARITY=?=True)" + "".join(
        f' && (Machine =!= "{m}")' for m in EXCLUDED_MACHINES)
    cluster = HTCondorCluster(
        cores=args.cores,
        processes=args.cores,
        memory=args.memory,
        disk=args.disk,
        python="python3",
        job_script_prologue=[f"source {LCG_VIEW}"],
        local_directory="$_CONDOR_SCRATCH_DIR",
        log_directory=os.path.abspath(args.log_dir),
        worker_extra_args=["--lifetime", args.lifetime, "--lifetime-stagger", "10m",
                           "--lifetime-restart"],
        job_extra_directives={
            "Requirements": requirements,
            "MY.SingularityImage": f'"{SINGULARITY_IMAGE}"',
            "+DESIRED_Sites": f'"{SITES}"',
            "+project_Name": '"cmssurfandturf"',
            "x509userproxy": f"/tmp/x509up_u{os.getuid()}",
            "use_x509userproxy": "True",
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT_OR_EVICT",
            "Stream_Output": "False",
            "Stream_Error": "False",
        },
        scheduler_options={"dashboard_address": f":{args.dashboard_port}"},
    )

    tmp = args.address_file + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(cluster.scheduler_address + "\n")
    os.replace(tmp, args.address_file)
    print(f"[dask_cluster] scheduler {cluster.scheduler_address}  dashboard {cluster.dashboard_link}  "
          f"{args.jobs} jobs x {args.cores} workers", flush=True)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    # Submit in batches: a single scale() fires every condor_submit at the schedd at once, and
    # dask_jobqueue does not retry one that times out.
    for n in [*range(SUBMIT_BATCH, args.jobs, SUBMIT_BATCH), args.jobs]:
        cluster.scale(jobs=n)
        deadline = time.monotonic() + 300
        while len(cluster.workers) < n and time.monotonic() < deadline and not stop.wait(1):
            pass
        if stop.is_set():
            break
    print(f"[dask_cluster] submitted {len(cluster.workers)} jobs", flush=True)

    while not stop.wait(5):
        pass

    print("[dask_cluster] shutting down, cancelling worker jobs", flush=True)
    cluster.close()
    try:
        os.remove(args.address_file)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
