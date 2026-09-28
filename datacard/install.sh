#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_16_0_0
cd CMSSW_16_0_0/src
cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v11.0.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b -j$(nproc --ignore=2) # always make a clean build, with n - 2 cores on the system