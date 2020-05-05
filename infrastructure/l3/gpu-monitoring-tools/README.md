# Helm charts for GPU metrics
To collect and visualize NVIDIA GPU metrics in kubernetes cluster, we have modified upstream prometheus-operator helm charts release-0.18. More information about the changes made are listed here.

## Identify and label GPU nodes
```
# Label GPU nodes to run our node-exporter only on GPU nodes.
# Note that a nodeSelector label is defined in node-exporter to control deploying it on GPU nodes only. 
kubectl label nodes <gpu-node-name> hardware-type=NVIDIAGPU
```

## Install helm charts
```
# Install helm https://docs.helm.sh/using_helm/ then run:
helm repo add gpu-helm-charts https://nvidia.github.io/gpu-monitoring-tools/helm-charts
helm repo update
helm install gpu-helm-charts/prometheus-operator --name prometheus-operator --namespace monitoring
helm install gpu-helm-charts/kube-prometheus --name kube-prometheus --namespace monitoring
```

## GPU metrics Dashboard
```
# Forward the port for Grafana.
kubectl -n monitoring port-forward $(kubectl get pods -n monitoring -lapp=kube-prometheus-grafana -ojsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}') 3000 &
# Open in browser http://localhost:3000 and go to Nodes Dashboard
```