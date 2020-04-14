from functools import reduce
import os
import yaml
import json
import re

def read_json(fp):
    with open(fp, 'r') as f:
        fs = f.read()
    return json.loads(fs)

def list_services(path):
    fn = os.path.join(path, '.serviceignore')
    print(fn)
    with open(fn, 'r') as f:
        lines = f.readlines()
    patterns = list(map(lambda x: x.rstrip(), filter(lambda x: x != '', lines)))
    files = []
    for r, d, f in os.walk(path):
        if 'data' in r:
            continue
        for file in f:
            if '.json' in file:
                if reduce(lambda x,y: x or y, map(lambda x: re.search(x, file) is not None, patterns)):
                    continue
                name = os.path.splitext(file)[0]
                try:
                    data = read_json(os.path.join(r, file))
                except:
                    print('Failed to read: ' + name)
                    exit(-1)
                data['container_name'] = name
                data['networks'] = {
                    'backendnet' : {}
                }
                files.append((name , data))
    return files

data = {
    'version' : '3.7',
    'services' : dict(list_services('services')),
    'networks' : {
        'backendnet' : {
            'name': 'backend'
        }
    }
}

with open('docker-compose.yaml', 'w') as f:
    f.write(yaml.dump(data))

os.system('docker-compose up --build > docker-compose.log')