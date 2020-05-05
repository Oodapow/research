# Why NFS ?

A Network File Server can be used as the storange component for a kubernetes cluster.

## How to setup a NFS ?

This can be done with the following command:
```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/nfs/os/ubuntu/main.sh | sudo bash -s -- /nfs/data 192.168.0.0/24
```