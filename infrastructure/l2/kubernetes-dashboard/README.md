# Kubernetes Dashboard

This is good tool for entrypoint cluster management. It offers the data that can be retreived by the `kubectl` tool in a Web UI.

## How to install the dathboard app ?

This is as simple as a `kubectl apply ...`:

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0/aio/deploy/recommended.yaml
```

## How to access the dashboard ?

First of all we need to make a service account that has admin rights in the cluster.

### Login credentials

To create the service account run the following command:

```
kubectl -n kube-system create serviceaccount k8-dashboard-sa
```

To give the role of cluster admin add the following role binding:

```
kubectl create clusterrolebinding k8-dashboard-cluster-admin --clusterrole=cluster-admin --serviceaccount=kube-system:k8-dashboard-sa
```

Now only the access token of this service account is needed to acces the dashboard:

```
kubectl -n kube-system get secret $(kubectl -n kube-system get serviceaccount/k8-dashboard-sa -o jsonpath='{.secrets[0].name}') -o jsonpath='{.data.token}' | base64 -d && echo
```

The above command can be understood like this:

  * `kubectl -n kube-system get serviceaccount/k8-dashboard-sa -o jsonpath='{.secrets[0].name}'` -> get token name
  * `kubectl -n kube-system get secret TOKEN_NAME -o jsonpath='{.data.token}'` -> get token secret
  * base64 decode of the token secret -> the token that can be used to login the dashboard

### Dashboard link

To obtain a dashboard link run the following command on your `kubectl` laptop.

```
kubectl proxy
```

This will make the dashboard avalable [HERE](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/)

