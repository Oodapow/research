curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/nfs/os/ubuntu/remove.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/nfs/os/ubuntu/apt-install.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/nfs/os/ubuntu/nfs-create-dict.sh | bash -s -- $1 && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/nfs/os/ubuntu/nfs-add-dict.sh | bash -s -- $1 $2 && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/nfs/os/ubuntu/nfs-save-config.sh | bash