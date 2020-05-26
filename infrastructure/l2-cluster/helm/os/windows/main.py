import os

if __name__ == '__main__':
    cmd = f'choco install kubernetes-helm --version={os.environ["HELM_VERSION"]}'
    print(cmd)
    os.system(cmd)
