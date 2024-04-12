#! /bin/bash
# adisve - 12-04-2024

VENV=".venv"
REQUIREMENTS="requirements.txt"
CONDA_ENV="thesis-project"

function make_directories() {
    echo "Creating directories .."
    mkdir -p out/
    mkdir -p out/{tensorboard,metrics,plots,server_measurements,tmp_model}
}

function initialize_python_env() {
    echo "Initializing python virtual environment .."
    python3 -m venv .venv
    source $VENV/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    while read requirement; do
        echo "Installing $requirement .."
        pip install "$requirement" > /dev/null 2>&1
    done < $REQUIREMENTS;
}

# Initializes ONLY CUDA and cudatoolkit using conda,
# not the whole python environment
function initialize_cuda_with_conda() {
    echo "Initializing CUDA with conda .."

    if [[ "$(uname -m)" == "x86_64" ]]; then
        ANACONDA_VERSION="Anaconda3-2024.02-1-Linux-x86_64"
    elif [[ "$(uname -m)" == "aarch64" ]]; then
        ANACONDA_VERSION="Anaconda3-2024.02-1-Linux-aarch64"
    elsee
        echo "Unsupported CPU architecture"
        exit 1
    fi

    wget https://repo.anaconda.com/archive/$ANACONDA_VERSION.sh

    bash $ANACONDA_VERSION.sh
    local shell="$SHELL"

    if [[ $shell == "/bin/bash" ]]; then
        echo "Initializing conda for bash"
        conda init bash || { echo "Error initializing conda for bash"; exit 1; }
    elif [[ $shell == "/bin/zsh" ]]; then
        echo "Initializing conda for zsh"
        conda init zsh || { echo "Error initializing conda for zsh"; exit 1; }
    else
        echo "Shell not supported"
        exit 1
    fi

    conda create -n $CONDA_ENV python=3.11;
    conda activate $CONDA_ENV;
    conda install -c nvcc cudatoolkit;
}

if [ ! -f "$REQUIREMENTS" ]; then
    echo "requirements.txt not found: Please create a requirements.txt file with the necessary packages."
    exit 1
fi

if [ ! -d "$VENV" ]; then
    initialize_python_env;
fi

if [ ! -d "out/" ]; then
    make_directories;
fi

if [[ "$(uname -s)" == "Linux" && -f "/etc/lsb-release" ]]; then
    source /etc/lsb-release
    if [[ "$DISTRIB_ID" == "Ubuntu" ]]; then
        initialize_cuda_with_conda;
    fi
fi