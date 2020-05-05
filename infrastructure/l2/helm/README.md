# Why helm ?

Helm is the package manager for kubernetes, it will be the main tool to use in setting up our machine learning cluster.

## Where to install helm ?

Helm should be installed on the administration machine (laptop).

## How to install helm ?

To install helm on any OS you can run (as administrator on windows) the following command:

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l2/helm/main.py | python
```

## How to init helm ?

Add service account and binding for `tiller`:

```
kubectl apply -f https://github.com/Oodapow/research/blob/master/infrastructure/l2/helm/yml/config.yml
```

Run the helm init command:

```
helm init --service-account tiller
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

Remove the curl alias!


Skip this step if you already have a Powershell profile:

```
New-Item $profile -force -itemtype file
```

Then edit your profile:

```
notepad $profile
```

Add the following line to it:

```
remove-item alias:curl
```

Save, close notepad and reload the profile with the command below or close and open Powershell to apply the profile:

```
. $profile
```