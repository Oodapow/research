curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/apt-install.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/edit-postgresql-conf.sh | bash -s -- $(find / -name "postgresql.conf" -print -quit)
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/edit-pghba-conf.sh | bash -s -- $(find / -name "pg_hba.conf" -print -quit) && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/restart-service.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/enable-service.sh | bash && \
curl -fsSL https://raw.githubusercontent.com/Oodapow/research/master/infrastructure/l1/postgresql/os/ubuntu/init-db.sh | bash -s -- $1