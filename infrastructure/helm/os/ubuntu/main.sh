HELM_FILE=helm-v$HELM_VERSION-linux-amd64.tar.gz

curl -O https://get.helm.sh/$HELM_FILE && \
tar -zxvf $HELM_FILE && \
mv linux-amd64/helm /usr/local/bin/helm && \
rm -rf $HELM_FILE linux-amd64