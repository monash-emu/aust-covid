#!/bin/bash

write_ios_s3 () {
   aws s3 cp $BASE_PATH/iodump s3://autumn-data/projects/aust_covid/alternate_analyses/2023-11-02T1102-vacc-d50k-t10k-b5k/.taskmeta/
}

export BASE_PATH=$PWD
cd code/autumn;
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate autumn310;
cd $BASE_PATH
git clone --branch revise-targets https://github.com/monash-emu/aust-covid.git
pip install -e ./aust-covid
echo Launching python task on projects/aust_covid/alternate_analyses/2023-11-02T1102-vacc-d50k-t10k-b5k
python -m autumn tasks springboard --run projects/aust_covid/alternate_analyses/2023-11-02T1102-vacc-d50k-t10k-b5k

if [ $? -ne 0 ]; then
    echo run python task failed, cleaning up
    echo FAILURE | aws s3 cp - s3://autumn-data/projects/aust_covid/alternate_analyses/2023-11-02T1102-vacc-d50k-t10k-b5k/.taskmeta/STATUS

    write_ios_s3
    sudo shutdown now
fi
echo Python task complete
write_ios_s3
sudo shutdown now
