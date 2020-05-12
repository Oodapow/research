apt update
apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

cd /tmp
wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tar.xz

tar -xf Python-3.7.2.tar.xz
cd Python-3.7.2
./configure --enable-optimizations

make -j 24
make install