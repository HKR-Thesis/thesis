#! /bin/bash
# adisve - 13-04-2024

PYTHON_VERSION="3.10"
CUDA_VERSION="11.0"
CONDA_ENV="cuda-enabled"

function install_conda_deps() {
    ./scripts/install-cuda.sh
    sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y > /dev/null 2>&1
}

function setup_conda_environment() {
    conda create -n $CONDA_ENV python=$PYTHON_VERSION;
    conda activate $CONDA_ENV;
    conda install nvidia::cuda-nvcc;
}

function initialize_cuda_with_conda() {
    echo "Initializing CUDA with conda .."

    install_conda_deps;

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b

    export PATH="/home/$(whoami)/miniforge3/bin:$PATH"
    shell_name=$(basename $SHELL)

    if [[ $shell_name == "bash" || $shell_name == "zsh" ]]; then
        echo "Initializing conda for $shell_name"
        conda init $shell_name || { echo "Error initializing conda for $shell_name"; exit 1; }
        source "$HOME/.${shell_name}rc" || { echo "Error sourcing ${shell_name}rc file"; exit 1; }
    else
        echo "Shell not supported"
        exit 1
    fi

    rm Miniforge3-$(uname)-$(uname -m).sh
}


if [ ! -d "/home/$(whoami)/miniforge3/" ]; then
        echo "Setting up Conda since this is not a Jetson device and Miniforge is not installed."
        initialize_cuda_with_conda
fi

if ! conda env list | grep "$CONDA_ENV" > /dev/null; then
    echo "Environment '$CONDA_ENV' does not exist. Creating now..."
    setup_conda_environment
fi