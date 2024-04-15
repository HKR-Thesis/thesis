#! /bin/bash
# adisve 14-04-2024

REQUIREMENTS_FILE=$1
VENV=".venv"

function install_venv_deps() {
    echo "Installing distro deps .."
    sudo apt install python3-venv -y > /dev/null 2>&1
}

function initialize_python_venv() {
    echo "Initializing python virtual environment .."
    python$PYTHON_VERSION -m venv $VENV
    source $VENV/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    while read requirement; do
        echo "Installing $requirement .."
        pip install "$requirement" > pip_logs.txt 2>&1
    done < $REQUIREMENTS_FILE;
}

install_venv_deps;
initialize_python_venv;