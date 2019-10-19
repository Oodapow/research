FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0 s3fs
RUN pip install opencv-python mlflow boto3 scikit-learn
RUN pip install matplotlib
ADD src /notebooks/mllib
WORKDIR /work
CMD ["bash"]
