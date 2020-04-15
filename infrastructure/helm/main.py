from sys import platform
import os

HELM_VERSION = '2.15.1'

if __name__ == '__main__':
    os.environ['HELM_VERSION'] = HELM_VERSION

    if platform == "linux" or platform == "linux2":
        os.system('')
    elif platform == "darwin":
        os.system('')
    elif platform == "win32":
        os.system('')