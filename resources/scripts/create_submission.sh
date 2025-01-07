#!/usr/bin/env bash

# delete tmp directory if it exists
if [ -d "tmp" ]; then
  rm -r tmp
fi

# create tmp directory
mkdir tmp

# get path of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

REPO_DIR=$DIR/../..

# copy the python package to the tmp directory
cp -r $REPO_DIR/hdr_hybrid_butterflies tmp/hdr_hybrid_butterflies

# copy models to the tmp directory
mkdir tmp/models
cp -r $REPO_DIR/data/models/segment_model tmp/models
cp -r $REPO_DIR/data/models/signal_hybrid_model tmp/models

# copy model.py to the tmp directory
cp $DIR/model.py tmp/model.py

# copy requirements.txt to the tmp directory
cp $DIR/requirements.txt tmp/requirements.txt

# copy tfscripts dependency to the tmp directory
cp -r /home/mhuennefeld/Repositories/github/TFScripts_kaggle/tfscripts tmp/tfscripts

# -------
# Testing
# -------
# make a test directory
mkdir tmp/test

# copy over some test images
mkdir tmp/test/images
mkdir tmp/test/images/hybrid
mkdir tmp/test/images/non-hybrid

cp $REPO_DIR/data/images/hybrid/CAM016784.jpg tmp/test/images/hybrid/
cp $REPO_DIR/data/images/non-hybrid/CAM041144.jpg tmp/test/images/non-hybrid/

# copy the metadata file to the test directory
cp $REPO_DIR/data/butterfly_anomaly_train.csv tmp/test/butterfly_anomaly_train.csv

# copy the test submission script to the test directory
cp $DIR/test_submission.py tmp/test/test_submission.py
# -------

# zip the contents of the tmp directory
cd tmp
zip -r submission.zip *

# move the zip file to the parent directory
cd ..
mv tmp/submission.zip .

# # delete the tmp directory
# rm -r tmp
