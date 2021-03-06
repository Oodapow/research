curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/apt-install.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/edit-postgresql-conf.sh | bash -s -- $(find /etc -name "postgresql.conf" -print -quit)
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/edit-pghba-conf.sh | bash -s -- $(find /etc -name "pg_hba.conf" -print -quit) && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/restart-service.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/enable-service.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1-machine/postgresql/os/ubuntu/init-db.sh | bash -s -- $1
