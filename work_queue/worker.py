#!/usr/bin/python3
import os
import json
import argparse

def load_defaults():
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defaults.json')
    print('\nLoading defaults from {}'.format(fn))
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    dd = load_defaults()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('container_name', type=str, help='')
    parser.add_argument('-b', '--build', action="store_true", help='')
    parser.add_argument('-d', '--detached', action="store_true", help='')
    parser.add_argument('-uri', '--rabbitmq_server_uri', default=dd['worker']['rabbitmq_server_uri'], type=str, help='')
    parser.add_argument('-u', '--rabbitmq_user', default=dd['worker']['rabbitmq_user'], type=str, help='')
    parser.add_argument('-p', '--rabbitmq_password', default=dd['worker']['rabbitmq_password'], type=str, help='')
    parser.add_argument('-q', '--rabbitmq_queue_name', default=dd['worker']['rabbitmq_queue_name'], type=str, help='')
    parser.add_argument('-gids', '--gpu_ids', type=str, default='-1', help='')

    args = parser.parse_args()

    tp = '--rm -it'
    if args.detached:
        tp = '-d --restart always'

    if args.build:
        os.system('docker build -t uipath/mlfarmer {}'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docker')))
        print('')

    template = 'docker run {} --network=backend -v /var/run/docker.sock:/var/run/docker.sock -e RABBITMQ_SERVER={} -e RABBITMQ_USER={} -e RABBITMQ_PASSWORD={} -e RABBITMQ_QUEUE_NAME={} -e NVIDIA_VISIBLE_DEVICES={} -e WORKER_NAME={} --name {} uipath/mlfarmer'
    cmd = template.format(tp, args.rabbitmq_server_uri, args.rabbitmq_user, args.rabbitmq_password, args.rabbitmq_queue_name, args.gpu_ids, args.container_name, args.container_name)
    print('> ' + cmd + '\n')
    os.system(cmd)