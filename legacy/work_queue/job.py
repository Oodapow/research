#!/usr/bin/python3
import os
import re
import sys
import json
import pika
import argparse

from mlflow.tracking import MlflowClient

from util import load_json, regex_type, abs_dir_type

def make_mlflow_client(args, mlflow, local=False):
    if local:
        return MlflowClient()
    else:
        print('')
        print('Connecting to MLFlow server: "{}"'.format(mlflow['server']['uri']))
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow['store']['s3_uri']
        os.environ['AWS_ACCESS_KEY_ID'] = mlflow['store']['s3_key']
        os.environ['AWS_SECRET_ACCESS_KEY'] = mlflow['store']['s3_secret']
        mlflow_client = MlflowClient(mlflow['server']['uri'])
        return mlflow_client

def build(args, registry):
    template = 'docker build -t {}/{} {}'
    cmd = template.format(registry['uri'], args.image_name, args.job_path)

    print('Building: "{}"'.format(args.image_name))
    print('')
    print('> ' + cmd)
    print('')
    os.system(cmd)
    print('')

def push(args, registry):
    template = 'docker push {}/{}'
    cmd = template.format(registry['uri'], args.image_name)

    print('Pushing: "{}" to "{}"'.format(args.image_name, registry['uri']))
    print('')
    print('> ' + cmd)
    print('')
    os.system(cmd)
    print('')

def run(args):
    currentDirectory = os.getcwd()
    os.chdir(os.path.join(args.job_path, 'src'))
    mlflow_client = make_mlflow_client(args, None, True)
    experiment = mlflow_client.get_experiment_by_name(args.image_name)
    print('Done!')
    print('')
    
    if experiment is None:
        eid = mlflow_client.create_experiment(args.image_name)
        print('Experiment with name: "{}" created.'.format(args.image_name))
        print('')
    else:
        eid = experiment.experiment_id
    
    run_id = mlflow_client.create_run(eid).info.run_id
    mlflow_client.log_artifacts(run_id, os.path.join(args.job_path, 'configs'), artifact_path='configs')

    template = '{} main.py --run-id {}'
    cmd = template.format(sys.executable, run_id)
    print('> ' + cmd)
    os.system(cmd)
    os.chdir(currentDirectory)

def queue(args, registry, mlflow, rabbitmq):
    mlflow_client = make_mlflow_client(args, mlflow)
    experiment = mlflow_client.get_experiment_by_name(args.image_name)
    print('Done!')
    print('')
    
    if experiment is None:
        eid = mlflow_client.create_experiment(args.image_name)
        print('Experiment with name: "{}" created.'.format(args.image_name))
        print('')
    else:
        eid = experiment.experiment_id
    
    mlflow['run']['id'] = mlflow_client.create_run(eid, tags={'mlflow.runName': mlflow['run']['name'], 'mlflow.user': mlflow['run']['user']}).info.run_id
    mlflow_client.set_tag(mlflow['run']['id'], 'queue_name', rabbitmq['queue_name'])

    mlflow_client.log_artifacts(mlflow['run']['id'], os.path.join(args.job_path, 'configs'), artifact_path='configs')

    print('')
    print('Connecting to RabbitMQ Server: "{}"'.format(rabbitmq['server_uri']))
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host = rabbitmq['server_uri'],
            credentials = pika.PlainCredentials(rabbitmq['user'], rabbitmq['password'])
        )
    )
    channel = connection.channel()
    print('Done!')
    print('')
    print('Queue to use: "{}"'.format(rabbitmq['queue_name']))
    print('')
    channel.queue_declare(queue=rabbitmq['queue_name'], durable=True)
    
    name, tag = args.image_name.split(':')

    data = {
        'image': args.image_name,
        'registry': registry,
        'mlflow': mlflow
    }

    message = json.dumps(data)
    channel.basic_publish(exchange='', routing_key=rabbitmq['queue_name'], body=message, properties=pika.BasicProperties(delivery_mode=2))
    connection.close()
    print('')
    print('Experiment scheduled!')
    print('')

if __name__ == '__main__':
    print('')
    mlflow = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'mlflow.json'))
    rabbitmq = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rabbitmq.json'))
    registry = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'registry.json'))

    parser = argparse.ArgumentParser()
    parser.add_argument('job_path', type=abs_dir_type, help='Path to a job.')
    parser.add_argument('image_name', type=regex_type(re.compile(r"[a-z]+:[a-z0-9]+$")), help='Name and tag for the image.')
    
    parser.add_argument('-b', '--build', help='', action="store_true")
    parser.add_argument('-p', '--push', help='', action="store_true")
    parser.add_argument('-r', '--run', help='', action="store_true")
    parser.add_argument('-q', '--queue', help='', action="store_true")
    
    args = parser.parse_args()
        
    if not (args.build or args.run or args.queue):
        raise argparse.ArgumentTypeError("Need to either build, run or queue.")
    
    if args.build:
        build(args, registry)

    if args.push:
        if not args.build:
            build(args, registry)
        push(args, registry)

    if args.run:
        run(args)
    elif args.queue:
        queue(args, registry, mlflow, rabbitmq)