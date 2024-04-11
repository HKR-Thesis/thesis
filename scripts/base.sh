#! /bin/bash
# adisve - 12-04-2024

VENV=".venv"
REQUIREMENTS="requirements.txt"

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

    wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
    bash Anaconda3-latest-Linux-x86_64.sh

    local shell="$SHELL"
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

    conda install -c nvcc cudatoolkit
}

if [ ! -d "$VENV" ]; then
    initialize_python_env;
fi

if [ ! -d "out/" ]; then
    make_directories;
fi