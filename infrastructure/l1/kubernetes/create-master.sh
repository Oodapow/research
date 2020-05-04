# pod-network-cidr must match the one from here -> https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/kubernetes/yml/calico.yml
sudo kubeadm init --pod-network-cidr=172.16.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl apply -f https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/kubernetes/yml/calico.yml
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta4/nvidia-device-plugin.yml