## Kubernetes NGINX Ingress

Installing this will allow the usage of the kubernetes `ingress` API object.

### How to install ?

Add helm repo:

```
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx/ 
```

Update helm charts:

```
helm repo update
```

Install chart:

```
helm install nginx-stable/nginx-ingress --name=nginx-ingress -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/ingress-nginx/yml/config.yml
```