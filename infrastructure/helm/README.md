# Why helm ?

Helm is the package manager for kubernetes, it will be the main tool to use in setting up our machine learning cluster.

## Where to install helm ?

Helm should be installed on the administration machine (laptop).

## How to install helm ?

To install helm on any OS you can run (as administrator on windows) the following command:

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/helm/main.py | python
```

### Windows specifics

To install on windows we need `curl` and `choco`

To install `choco` run the following command in an administrator poweshell:
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

To install the linux curl run the following command:
```
choco install curl
```

To use the linux curl it s needed to disable the poweshell alias like this:
```
Remove-item alias:curl
```