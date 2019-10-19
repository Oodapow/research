import argparse
import json
import os

def load_json(fn):
    print('Loading json from {}'.format(fn))
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def regex_type(patern):
    def _inner(s):
        if not patern.match(s):
            raise argparse.ArgumentTypeError('Not matching: "{}"'.format(patern))
        return s
    return _inner

def abs_dir_type(s):
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError('Not a dir: {}'.format(s))
    return os.path.abspath(s)