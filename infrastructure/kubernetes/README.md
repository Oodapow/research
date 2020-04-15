# Requirements

To install the tools needed to deploy the kubernetes platform you need the following:
 * 2 CPUs
 * 2 GB RAM
 * swap off

## How to turn off swap?

Run the following command:

```
sudo swapoff -a
```

Edit the file `/etc/fstab` to comment the swap line:

```
sudo vi /etc/fstab
```

Check swap partitions:

```
sudo blkid | grep swap  
```

Mask swap partitions:

```
sudo systemctl mask dev-sd**.swap
```

Reboot the system:

```
sudo reboot
```

## How to install kubernetes req ?

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/kubernetes/os/ubuntu/main.sh | sudo sh
```

## How to setup kubernetes master ?

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/kubernetes/create-master.sh | sh
```

To watch the status of the master pods:

```
kubectl get pods --all-namespaces --watch
```

After `kube-apiserver-master`, `kube-controller-manager-master`, `kube-scheduler-master` and `etcd-master` are in the running state you can check the nodes like this:

```
kubectl get nodes
```

## How to connect a node to the cluster ?

```
kubeadm join 192.168.0.109:6443 --token TOKEN_HERE \
    --discovery-token-ca-cert-hash sha256:SHA_HERE
```

To find the cmd line look in the output of the `create-master.sh` script on the master node.

```
#If you didn't keep the output, on the master, you can get the token.
kubeadm token list

#If you need to generate a new token, perhaps the old one timed out/expired.
kubeadm token create

#On the master, you can find the ca cert hash.
openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | openssl rsa -pubin -outform der 2>/dev/null | openssl dgst -sha256 -hex | sed 's/^.* //'
```

Wait for the nodes to enter the ready state:

```
kubectl get nodes --watch
```