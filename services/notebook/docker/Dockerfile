FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3-jupyter
RUN pip install opencv-python jupyterlab pika minio
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0

RUN apt-get update
RUN apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
RUN apt-get update
RUN apt-get install -y docker-ce docker-ce-cli containerd.io

RUN curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
RUN chmod +x /usr/local/bin/docker-compose

RUN pip install mlflow

RUN apt-get install -y s3fs

RUN pip install boto3

EXPOSE 8888
WORKDIR /notebooks

CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0"]
