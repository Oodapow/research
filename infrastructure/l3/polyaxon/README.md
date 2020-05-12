# How to set up the platform ?

To set up the Polyaxon platform we need to do the following:
 * ensure persistence
 * install with helm

## Create the PVC

To create the PVC needed for the platform run the following command:

```
kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3/polyaxon/yml/persist.yml
```

## Install with Helm

To install polyaxon with helm run the follwing command:

```
 helm install polyaxon/polyaxon --name=polyaxon --namespace=polyaxon -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3/polyaxon/yml/config.yml
```