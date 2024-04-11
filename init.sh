#! /bin/bash
# adisve - 12-04-2024

make_directories()
initialize_python_env()

if [[ $(lsb_release -si) == "Ubuntu" ]]; then
    initialize_cuda_with_conda()
else
    echo "Conda initialization is only compatible with Ubuntu."
    exit 1
fi

make_directories() {
    mkdir -p out/
    mkdir -p out/{logs,metrics,plots,server_measurements,tmp_model}
}

initialize_python_env() {
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
}

# Initializes ONLY CUDA and cudatoolkit using conda,
# not the whole python environment
initialize_cuda_with_conda() {
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