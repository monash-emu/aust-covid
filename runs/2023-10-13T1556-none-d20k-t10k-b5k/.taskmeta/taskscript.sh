#!/bin/bash

write_ios_s3 () {
   aws s3 cp $BASE_PATH/iodump s3://autumn-data/projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k/.taskmeta/
}

export BASE_PATH=$PWD
cd code/autumn;
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"; conda activate autumn310;
cd $BASE_PATH
git clone --branch finalise-notebooks https://github.com/monash-emu/aust-covid.git

if [ $? -ne 0 ]; then
    echo git clone --branch finalise-notebooks https://github.com/monash-emu/aust-covid.git failed, cleaning up
    echo FAILURE | aws s3 cp - s3://autumn-data/projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k/.taskmeta/STATUS

    write_ios_s3
    sudo shutdown now
fi
pip install -e ./aust-covid

if [ $? -ne 0 ]; then
    echo pip install -e ./aust-covid failed, cleaning up
    echo FAILURE | aws s3 cp - s3://autumn-data/projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k/.taskmeta/STATUS

    write_ios_s3
    sudo shutdown now
fi
echo Launching python task on projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k
python -m autumn tasks springboard --run projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k

if [ $? -ne 0 ]; then
    echo run python task failed, cleaning up
    echo FAILURE | aws s3 cp - s3://autumn-data/projects/aust_covid/alternate_analyses/2023-10-13T1556-none-d20k-t10k-b5k/.taskmeta/STATUS

    write_ios_s3
    sudo shutdown now
fi
echo Python task complete
write_ios_s3
sudo shutdown now
