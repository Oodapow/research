from sys import platform
import os

HELM_VERSION = '2.15.1'

if __name__ == '__main__':
    os.environ['HELM_VERSION'] = HELM_VERSION

    if platform == "linux" or platform == "linux2":
        print('Running Ubuntu Setup')
        os.system('curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm/os/ubuntu/main.sh | sudo sh')
    elif platform == "darwin":
        print('Running macOS Setup')
        os.system('curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm//os/mac/main.sh | sudo sh')
    elif platform == "win32":
        print('Running Windows Setup')
        os.system('curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm/os/windows/main.py | python')