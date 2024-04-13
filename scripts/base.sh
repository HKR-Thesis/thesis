#! /bin/bash
# adisve - 12-04-2024

VENV=".venv"
OUTPUT_DIR="out"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.0"
REQUIREMENTS="requirements.txt"
CONDA_ENV="thesis-project"

function install_venv_deps() {
    echo "Installing distro deps .."
    sudo apt install python3-venv -y > /dev/null 2>&1
}

function is_jetson() {
    if [ -f "/etc/nv_tegra_release" ]; then
        return 0
    else
        return 1
    fi
}

function install_conda_deps() {
    ./scripts/install-cuda.sh
    sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y > /dev/null 2>&1
}

function make_directories() {
    echo "Creating directories .."
    mkdir -p $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR/{tensorboard,metrics,plots,server_measurements,tmp_model}
}

function initialize_python_venv() {
    echo "Initializing python virtual environment .."
    python$PYTHON_VERSION -m venv $VENV
    source $VENV/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    while read requirement; do
        echo "Installing $requirement .."
        pip install "$requirement" > /dev/null 2>&1
    done < $REQUIREMENTS;
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


    # Cleanup
    rm Miniforge3-$(uname)-$(uname -m).sh
}

function setup_conda_environment() {
    conda create -n $CONDA_ENV python=$PYTHON_VERSION;
    conda activate $CONDA_ENV;
    conda install cudatoolkit=$CUDA_VERSION -c nvidia;
}

if [ ! -f "$REQUIREMENTS" ]; then
    echo "requirements.txt not found: Please create a requirements.txt file with the necessary packages."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    make_directories;
fi

# CUDA on Jetson devices should be handled specifically
# by the distribution, not by Conda
if ! is_jetson; then
    if [ ! -d "/home/$(whoami)/miniforge3/" ]; then
        echo "Setting up Conda since this is not a Jetson device and Miniforge is not installed."
        initialize_cuda_with_conda
    fi

    if ! conda env list | grep "$CONDA_ENV" > /dev/null; then
        echo "Environment '$CONDA_ENV' does not exist. Creating now..."
        setup_conda_environment
    fi
else
    echo "This is a Jetson device. Skipping all Conda-related setup."
fi

if [ ! -d "$VENV" ]; then
    install_venv_deps;
    initialize_python_venv;
fi
