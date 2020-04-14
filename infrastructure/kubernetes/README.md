# Requirements

To install the tools needed to deploy the kubernetes platform you need the following:
 * 2 CPUs
 * 2 GB RAM
 * swap off

## OS versions

 * the scripts in this folder work only on `Ubuntu 16.04` server or desktop.

## How to turn off swap?

Run the following command:

```
sudo swapoff -a
```

Edit the file `/etc/fstab` to comment the swap line:

```
sudo vi /etc/fstab
```

Reboot the system:

```
sudo reboot
```