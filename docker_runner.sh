# docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow bash

cd /home/klsvd/laboro/NextGIS/AlarmArkh
docker build tensorflow.rs -t tensorflow.rs

docker run -u $(id -u):$(id -g)  --gpus all -it --rm  -v /mnt/alpha/backups/alarm/ARKH_TRAIN_DATA:/data/Alarm/Samples  -v $PWD:/tmp -w /tmp tensorflow.rs bash

python3 LandsatToSentinel/ls2snt.py 


