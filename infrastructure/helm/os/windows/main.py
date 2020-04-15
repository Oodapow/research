import os

if __name__ == '__main__':
    os.system(f'choco install kubernetes-helm --version={os.environ["HELM_VERSION"]}')
