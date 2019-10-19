#!/usr/bin/python3
import os
import json
import argparse
from util import load_json

if __name__ == '__main__':
    print('')
    rabbitmq = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'rabbitmq.json'))

    parser = argparse.ArgumentParser()
    parser.add_argument('container_name', type=str, help='')
    parser.add_argument('-b', '--build', action="store_true", help='')
    parser.add_argument('-d', '--detached', action="store_true", help='')
    parser.add_argument('-gids', '--gpu_ids', type=str, default='-1', help='')

    args = parser.parse_args()

    tp = '--rm -it'
    if args.detached:
        tp = '-d --restart always'

    if args.build:
        os.system('docker build -t mlfarm/worker {}'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docker')))
        print('')

    template = 'docker run {} --network=backend -v /var/run/docker.sock:/var/run/docker.sock -e RABBITMQ_SERVER={} -e RABBITMQ_USER={} -e RABBITMQ_PASSWORD={} -e RABBITMQ_QUEUE_NAME={} -e NVIDIA_VISIBLE_DEVICES={} -e WORKER_NAME={} --name {} mlfarm/worker'
    cmd = template.format(tp, rabbitmq['server_uri'], rabbitmq['user'], rabbitmq['password'], rabbitmq['queue_name'], args.gpu_ids, args.container_name, args.container_name)
    print('> ' + cmd + '\n')
    os.system(cmd)