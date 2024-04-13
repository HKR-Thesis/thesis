#! /bin/bash
# adisve - 12-04-2024

VENV=".venv"
REQUIREMENTS="requirements.txt"
CONDA_ENV="thesis-project"

function install_venv_deps() {
    echo "Installing distro deps .."
    sudo apt install python3-venv -y > /dev/null 2>&1
}

function install_conda_deps() {
    sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y > /dev/null 2>&1
}

function make_directories() {
    echo "Creating directories .."
    mkdir -p out/
    mkdir -p out/{tensorboard,metrics,plots,server_measurements,tmp_model}
}

function initialize_python_env() {
    echo "Initializing python virtual environment .."
    python3 -m venv $VENV
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

    install_conda_deps;

    if [[ "$(uname -m)" == "x86_64" ]]; then
        ARCHITECTURE="x86_64"
    elif [[ "$(uname -m)" == "aarch64" ]]; then
        ARCHITECTURE="aarch64"
    else
        echo "Unsupported CPU architecture"
        exit 1
    fi

    wget http://repo.continuum.io/miniconda/Miniconda3-py39_4.9.2-Linux-$ARCHITECTURE.sh

    bash Miniconda3-py39_4.9.2-Linux-$ARCHITECTURE.sh -b
    local shell="$SHELL"

    export PATH="/home/$(whoami)/miniconda3/bin:$PATH"

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
    
    if [[ $shell == "/bin/bash" ]]; then
	source $HOME/.bashrc;
    elif [[ $shell == "/bin/zsh" ]]; then
	source $HOME/.zshrc;
    else
	echo "Shell not supported .."
	exit 1
    fi

    # Create conda env and install cuda stuff
    conda create -n $CONDA_ENV python=3.11;
    conda activate $CONDA_ENV;
    conda install cuda cudatoolkit -c nvidia;
}

if [ ! -f "$REQUIREMENTS" ]; then
    echo "requirements.txt not found: Please create a requirements.txt file with the necessary packages."
    exit 1
fi

if [ ! -d "$VENV" ]; then
    install_venv_deps;
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
