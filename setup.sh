#!/bin/bash

# Download anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# Install anaconda
shasum -a 256 Anaconda3*.sh
bash Anaconda3*.sh

echo "Installed Anaconda; run 'source ~/.bashrc; conda env create -f environment.yml' to complete setup."
echo "Make sure to update git config --global user.name and user.email."
echo "To train GPT, download and extract the OpenWebText dataset into the './data' subdirectory."

# Delete install scripts
rm Anaconda3*.sh