# Machine Learning Research

Here is a repo that offers a way to develop ML in an easy to set up environment.

## Requirements

To be able to research you need a machine with:
 * CUDA and cuDNN
 * Python3
 * Docker
 * NVIDIA Docker
 * Docker Compose

## What can you work with ?

Here you can start a suite of services in containers that can be usefull in ML pipelines:
 * RabbitMQ -> queue system for managing the experiment queues
 * MLFlow -> expermiment tracking with S3 artifacts store
 * Minio -> open source S3 implementation
 * Docker Registry -> to keep the docker images of the experiments 
 * Jupyter Lab -> ofers a notebook interface that has acces to docker on the host machine

## How to run ?

```
sudo python docker_composer.py
```

This will build the images and start the services.


To access the notebook go to `IP:8888` in your browser.
