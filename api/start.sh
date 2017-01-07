#!/bin/bash

set -e -u

cd ../openface/openface/ # Root OpenFace directory.

LOG='/tmp/openface.flask.api.log'
export FLASK_APP=../../api/api.py
flask run -p 5000 -h 0.0.0.0 --with-threads 2>&1
# | tee $LOG &
# --debugger
