FROM pytorch/pytorch

RUN apt-get update
RUN apt-get -y install g++
RUN pip install numpy==1.24.3 scipy==1.11.1 librosa==0.9.2 pesq==0.0.4 tensorboardX==2.6.2 h5py==3.9.0 matplotlib==3.7.2
WORKDIR /segan/
COPY . .
RUN apt-get -y install wget
RUN wget http://veu.talp.cat/seganp/release_weights/segan+_generator.ckpt -P ckpt_segan+