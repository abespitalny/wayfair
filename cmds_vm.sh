#!/bin/bash

# install python 3.7
cd /usr/src
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev \
libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
sudo wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz
sudo tar xzf Python-3.7.3.tgz
sudo rm Python-3.7.3.tgz
cd Python-3.7.3
sudo ./configure --enable-optimizations
sudo make altinstall

# check installation was successful
p_v=$(python3.7 -V)
if [["$p_v" == *3.7.3*]]; then
    echo "Python successfully installed"
else
    echo "Better luck next time :("
    exit 1
fi

# install necessary libraries for python
sudo python3.7 -m pip install scikit-learn

# get repo
sudo apt-get install git
cd ~
git clone https://github.com/abespitalny/wayfair.git

# download pm2
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install pm2 -g


