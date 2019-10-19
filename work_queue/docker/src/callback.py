import os
import sys
import json
import shlex
import mlflow
from subprocess import Popen, PIPE
from threading  import Thread

def tee(infile, *files):
    def fanout(infile, *files):
        for line in iter(infile.readline, ''):
            if not line:
                break
            for f in files:
                f.write(line.decode("utf-8"))
                f.write('\n')
        infile.close()
    t = Thread(target=fanout, args=(infile,)+files)
    t.daemon = True
    t.start()
    return t

def teed_call(cmd_args, **kwargs):    
    stdout, stderr = [kwargs.pop(s, None) for s in ['stdout', 'stderr']]
    p = Popen(cmd_args,
              stdout=PIPE if stdout is not None else None,
              stderr=PIPE if stderr is not None else None,
              **kwargs)
    threads = []
    if stdout is not None: threads.append(tee(p.stdout, stdout, sys.stdout))
    if stderr is not None: threads.append(tee(p.stderr, stderr, sys.stderr))
    return threads, p.wait()

def callback(ch, method, properties, body):
    print(100*'+')
    print('')
    try:
        data = json.loads(body.decode("utf-8"))
        image, registry, mlflow = data['image'], data['registry'], data['mlflow']
    except:
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(body.decode("utf-8"))
        print('')
        print('Invalid Message!')
        print('')
        print(100*'-')
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        return
    print('')
    print('Connecting to MLFlow server: "{}"'.format(mlflow['server']['uri']))
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = mlflow['store']['s3_uri']
    os.environ['AWS_ACCESS_KEY_ID'] = mlflow['store']['s3_key']
    os.environ['AWS_SECRET_ACCESS_KEY'] = mlflow['store']['s3_secret']
    mlflow_client = mlflow.tracking.MlflowClient(mlflow['server']['uri'])
    experiment = mlflow_client.get_experiment_by_name(image)
    print('Done!')
    print('')
    
    runtime = ''
    if os.environ['NVIDIA_VISIBLE_DEVICES'] != '-1':
        runtime = ' --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES='+os.environ['NVIDIA_VISIBLE_DEVICES']
    template = 'docker run{} --privileged --network=backend --rm {}/{}'
    cmd = template.format(runtime, registry['uri'], image)
    print('> ' + cmd)
    print('')
    
    fout, ferr = open('out.log', 'w'), open('err.log', 'w')
    threads, exitcode = teed_call(shlex.split(cmd), stdout=fout, stderr=ferr)
    for t in threads: t.join()
    fout.close()
    ferr.close()
    
    mlflow_client.log_artifact(mlflow['run']['id'], 'out.log')
    if exitcode is not 0:
        mlflow_client.log_artifact(mlflow['run']['id'], 'err.log')

    print(100*'-')
    if exitcode is 0:
        mlflow_client.set_terminated(mlflow['run']['id'])
        os.environ['AWS_ACCESS_KEY_ID'] = ''
        os.environ['AWS_SECRET_ACCESS_KEY'] = ''
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = ''
        ch.basic_ack(delivery_tag=method.delivery_tag)
    else:
        mlflow_client.set_terminated(mlflow['run']['id'], 'FAILED')
        os.environ['AWS_ACCESS_KEY_ID'] = ''
        os.environ['AWS_SECRET_ACCESS_KEY'] = ''
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = ''
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)