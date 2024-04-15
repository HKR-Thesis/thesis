#!/bin/bash

function install_cuda() {
    if command -v nvcc &> /dev/null
    then
        echo "CUDA is already installed."
    else
        echo "CUDA not found. Installing CUDA..."

        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/cuda-ubuntu1804.pin
        sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

        wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
        sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
        sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
        sudo apt update
        sudo apt -y install cuda

        echo "CUDA installation completed."

        # Update path
        set_path_variables

	# Cleanup
	rm cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
    fi
}

function set_path_variables() {
    echo "Setting path variables..."
    shell_name=$(basename $SHELL)

    if [[ $shell_name == "bash" || $shell_name == "zsh" ]]; then
        echo 'export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}' >> ~/.${shell_name}rc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.${shell_name}rc

        source ~/.${shell_name}rc
    else
        echo "Shell not supported"
        exit 1
    fi

    echo "Path variables set."
}

#install_cuda
set_path_variables
