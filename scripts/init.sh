#! /bin/bash
# adisve - 12-04-2024


OUTPUT_DIR="out"
VENV=".venv"


function is_jetson() {
    if [ -f "/etc/nv_tegra_release" ]; then
        return 0
    else
        return 1
    fi
}

function make_directories() {
    echo "Creating directories .."
    mkdir -p $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR/{tensorboard,metrics,plots,server_measurements,tmp_model,figures}
}

if [ ! -d "$OUTPUT_DIR" ]; then
    make_directories;
fi

if ! is_jetson; then
    ./scripts/conda.sh
else
    echo "This is a Jetson device. Skipping all Conda-related setup."
fi

if [ ! -d "$VENV" ]; then
    if is_jetson; then
        ./scripts/venv.sh "jetson-requirements.txt"
    else
        ./scripts/venv.sh "server-requirements.txt"
    fi
fi
