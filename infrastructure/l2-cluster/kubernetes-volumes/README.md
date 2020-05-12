# K8s PV Provisioner

This is the way in witch a persistent volume claim can obtain a dynamicaly allocated persistent volume.

## How to set it up ?

First we need to create a service account with the right acces; this can be done with the following command:

```
kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l2-cluster/kubernetes-volumes/yml/rbac.yml
```

The we need to create a default storage class to be used by the claims:

```
kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l2-cluster/kubernetes-volumes/yml/sc.yml
```

In the end we need to start the provisioner. To do this download [THIS](https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l2-cluster/kubernetes-volumes/yml/nfs-prov.yml) file and edit the IP of the NFS server. To apply it run the command:

```
kubectl apply -f nfs-prov.yml
```

To check the install is working run the following command:

```
kubectl get pods --watch
```

Wait until the `nfs-pod-provisioner-*` pod is in the `Running` state