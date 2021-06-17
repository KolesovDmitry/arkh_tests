# docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow bash

cd /home/klsvd/laboro/NextGIS/AlarmArkh
docker build tensorflow.rs -t tensorflow.rs

sudo docker run -u $(id -u):$(id -g)  --gpus all -it --rm  -v $PWD:/tmp -w /tmp tensorflow.rs bash
