#!/bin/bash

# update packages
sudo apt update
sudo apt -y upgrade

# install python
sudo apt -y install python3 python3-pip

# install OpenCV using pip
pip3 install -I opencv-contrib-python==3.4.5.20
