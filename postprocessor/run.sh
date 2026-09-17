#!/bin/bash

OUTPUT_DIR=/ceph/cms/store/user/aaarora/vbsvvh/postprocessing/latest/
CUTFLOW_DIR=cutflows
mkdir -p ${CUTFLOW_DIR}
# preselection condor logs (for the preselection half of the combined cutflow) default to
# <repo>/preselection/condor/jobs; override with --cutflow-logs <dir> if they live elsewhere.

# # sig
nohup python3 postprocess.py --channel 1lep_2FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_2fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_2fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r2_1lep_2fj_sig.log &
nohup python3 postprocess.py --channel 1lep_2FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_2fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_2fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r3_1lep_2fj_sig.log &

nohup python3 postprocess.py --channel 1lep_1FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_1fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_1fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r2_1lep_1fj_sig.log &
nohup python3 postprocess.py --channel 1lep_1FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_1fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_1fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r3_1lep_1fj_sig.log &

nohup python3 postprocess.py --channel 0lep_3FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_0lep_3fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_0lep_3fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r2_0lep_3fj_sig.log &
nohup python3 postprocess.py --channel 0lep_3FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_0lep_3fj_sig*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_0lep_3fj_sig.txt --cutflow-split 'C2V_[^_/]+_C3_[^_/]+' >r3_0lep_3fj_sig.log &


# # data
nohup python3 postprocess.py --channel 1lep_2FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_2fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_2fj_data.txt > r2_1lep_2fj_data.log &
nohup python3 postprocess.py --channel 1lep_2FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_2fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_2fj_data.txt > r3_1lep_2fj_data.log &

nohup python3 postprocess.py --channel 1lep_1FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_1fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_1fj_data.txt > r2_1lep_1fj_data.log &
nohup python3 postprocess.py --channel 1lep_1FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_1fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_1fj_data.txt > r3_1lep_1fj_data.log &

nohup python3 postprocess.py --channel 0lep_3FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_0lep_3fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_0lep_3fj_data.txt > r2_0lep_3fj_data.log &
nohup python3 postprocess.py --channel 0lep_3FJ --threads 16 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_0lep_3fj_data*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_0lep_3fj_data.txt > r3_0lep_3fj_data.log &


# # bkg
nohup python3 postprocess.py --channel 1lep_2FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_2fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_2fj_bkg.txt > r2_1lep_2fj_bkg.log &
nohup python3 postprocess.py --channel 1lep_2FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_2fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_2fj_bkg.txt > r3_1lep_2fj_bkg.log &

nohup python3 postprocess.py --channel 1lep_1FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_1lep_1fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_1lep_1fj_bkg.txt > r2_1lep_1fj_bkg.log &
nohup python3 postprocess.py --channel 1lep_1FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_1lep_1fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_1lep_1fj_bkg.txt > r3_1lep_1fj_bkg.log &

nohup python3 postprocess.py --channel 0lep_3FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r2_0lep_3fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r2_0lep_3fj_bkg.txt > r2_0lep_3fj_bkg.log &
nohup python3 postprocess.py --channel 0lep_3FJ --threads 32 --input /ceph/cms/store/user/aaarora/vbsvvh/preselection/latest/*r3_0lep_3fj_bkg*/*/output_*.root --output-dir ${OUTPUT_DIR} --cutflow ${CUTFLOW_DIR}/r3_0lep_3fj_bkg.txt > r3_0lep_3fj_bkg.log &
