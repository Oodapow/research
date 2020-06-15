# Kubernetes NGINX Ingress

Installing this will allow the usage of the kubernetes `ingress` API object.

## Manage with Helm

Update helm charts:

```
helm repo update
```

Install chart:

```
helm install stable/nginx-ingress --name=nginx-ingress -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/ingress-nginx/yml/config.yml
```

Update chart:

```
helm upgrade nginx-ingress stable/nginx-ingress -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l3-app/ingress-nginx/yml/config.yml
```

Uninstall chart:
```
helm del --purge nginx-ingress
```

Status:
```
helm status nginx-ingress
```