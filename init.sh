#! /bin/bash
# adisve - 12-04-2024

VENV=".venv"
REQUIREMENTS="requirements.txt"

function make_directories() {
    mkdir -p out/
    mkdir -p out/{logs,metrics,plots,server_measurements,tmp_model}
}

function initialize_python_env() {
    python3 -m venv .venv
    source $VENV/bin/activate
    while read requirement; do pip install "$requirement"; done < $REQUIREMENTS;
}

# Initializes ONLY CUDA and cudatoolkit using conda,
# not the whole python environment
function initialize_cuda_with_conda() {
    wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
    bash Anaconda3-latest-Linux-x86_64.sh

    shell=$(SHELL)
    if [[ $shell == "/bin/bash" ]]; then
        echo "Initializing conda for bash"
        conda init bash
    elif [[ $shell == "/bin/zsh" ]]; then
        echo "Initializing conda for zsh"
        conda init zsh
    else
        echo "Shell not supported"
        exit 1
    fi

    conda create -n thesis-project python=3.11
    conda activate thesis-project

    conda install -c anaconda nvcc cudatoolkit
}


if [ ! -d "$VENV" ]; then
    initialize_python_env
fi

if [ ! -d "out/" ]; then
    make_directories
fi

if [[ $(lsb_release -si 2> /dev/null) == "Ubuntu" ]]; then
    initialize_cuda_with_conda
else
    echo "Conda initialization using this script is only compatible with Ubuntu."
    exit 1
fi