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

## How to install kubernetes prerequisites ?

This is to be run on all the cluster nodes, master and workers alike.

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/kubernetes/os/ubuntu/main.sh | sudo sh
```

For nodes with GPUs, the following command is also needed:

```
curl -fsSL https://raw.githubusercontent.com/UiPath/Infrastructure/master/ML/prereq_installer.sh | sudo bash
```

## How to setup kubernetes master ?

```
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/kubernetes/create-master.sh | sh
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
kubeadm join MASTER_IP:6443 --token TOKEN_HERE \
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

## How to test GPUs in the cluster ?

To start a pod with GPU access run the following:

```
kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/kubernetes/tests/gpu-access.yml
```

Wait for the pod `cuda-vector-add` to enter in the `Completed` state:

```
kubectl get pods --watch
```

To check the output log:

```
kubectl logs cuda-vector-add
```

If all works well the pod can be deleted like this:

```
kubectl delete pod cuda-vector-add
```

## How to use kubectl on any machine ?

`kubectl` also be used on a laptop. To do that use this command on the administrative laptop:

```
scp ADMIN_USER@MASTER_IP:/home/ADMIN_USER/.kube/config $HOME/.kube/config
```

Replace `ADMIN_USER` and `MASTER_IP` with the correct values.

## How to remove node from cluster ?

List node and watch for the target name:
```
kubectl get nodes
```

Drain the node (stop assigning pods to that node and remove current ones):
```
kubectl drain NODE_NAME
```

Remove node from the cluster:
```
kubectl delete node NODE_NAME
```
