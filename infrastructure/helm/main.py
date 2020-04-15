from sys import platform
from subprocess import Popen, PIPE
import os

HELM_VERSION = '2.15.1'

def run_script(url, shell):
    with Popen(['curl', '-fsSL', url], stdout=PIPE, env=os.environ) as curl:
        with Popen(['sudo', shell], stdin=curl.stdout, stdout=PIPE, env=os.environ) as shell:
            curl.stdout.close()
            curl.wait()
            shell.wait()

if __name__ == '__main__':
    os.environ['HELM_VERSION'] = HELM_VERSION

    url = None
    shell = 'sh'

    if platform == "linux" or platform == "linux2":
        print('Running Ubuntu Setup')
        url = 'https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm/os/ubuntu/main.sh'
    elif platform == "darwin":
        print('Running macOS Setup')
        url = 'https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm//os/mac/main.sh'
    elif platform == "win32":
        print('Running Windows Setup')
        url = 'https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm/os/windows/main.py'
        shell = 'python'
    
    run_script(url, shell)