# How to set up the platform ?

To set up the Polyaxon platform we need to do the following:
 * ensure persistence
 * install with helm

## Create the PVC

To create the PVC needed for the platform run the following command:

```
kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/polyaxon/yml/persist.yml -n polyaxon
```

## Create TLS Secret

Copy files:

```
scp .\ca_bundle.crt oodapow@192.168.0.108:/home/oodapow/tls/polyaxon/ca_bundle.crt
scp .\certificate.crt oodapow@192.168.0.108:/home/oodapow/tls/polyaxon/certificate.crt
scp .\private.key oodapow@192.168.0.108:/home/oodapow/tls/polyaxon/private.key
```

Merge files on server:
```
for f in certificate.crt ca_bundle.crt; do (cat "${f}"; echo) >> mcertificate.crt; done
```

```
kubectl create secret tls mlfarm-tls -n polyaxon --key private.key --cert mcertificate.crt
```

## Manage with Helm

Add helm repo:

```
helm repo add polyaxon https://charts.polyaxon.com
```

Update helm charts:

```
helm repo update
```

To install polyaxon with helm run the follwing command:

```
helm install polyaxon/polyaxon --name=polyaxon --namespace=polyaxon -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/polyaxon/yml/config.yml
```

Update chart:

```
helm upgrade polyaxon polyaxon/polyaxon -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/polyaxon/yml/config.yml
```

Uninstall chart:
```
helm del --purge polyaxon
```

Status:
```
helm status polyaxon
```

## Issues

Here is a list of workaround that one might need when setting up the polyaxon platform.

### Can't login

This can happen because for some reason the user is not added to the database.


To solve this you need to start an interactive shell in the polyaxon-api container of the polyaxon api pod and use a polyaxon utility script to add the superuser in the database.


Run the following to get the pod info:
```
kubectl get pods -lrole=polyaxon-api -n polyaxon
```

this should output something like this:
```
NAME                                     READY   STATUS    RESTARTS   AGE
polyaxon-polyaxon-api-6c8c54fc6c-4s5wf   2/2     Running   0          60m
```

Run the following to start an interactive shell in the polyaxon-api container of the polyaxon api pod:
```
kubectl exec -it POLYAXON_API_POD_NAME -c polyaxon-api -n polyaxon -- /bin/bash
```

Now that you are inside the api container run the following script to add the user:
```
python polyaxon/manage.py createuser --superuser --username USERNAME --password PASSWORD --email EMAIL
```