# About

This project is a simple example of training a pendulum to balance using Q learning. The pendulum is a simple inverted pendulum system with a single joint and a mass at the end. The pendulum is trained using a Q learning algorithm to balance the pendulum in the upright position.


https://github.com/HKR-Thesis/traininig_environment_pendulum/assets/89925489/8556d586-fc47-44ae-9b20-d89ca00a4ce2

# Installation

The project provides several setup scripts under scripts/ to install the necessary dependencies. An assumption before initializing or running for the Jetson Nano is that tensorflow has already been setup on the device (instructions can be found below). The scripts are as follows:

- init.sh: Creates a general setup of the project
- conda.sh: Installs the conda environment if the underlying device is not a Jetson Nano
- cuda.sh: Installs the necessary cuda dependencies for devices that are not Jetson Nano
- venv.sh : Installs the necessary python dependencies for the project based on the underlying device


## OKDO Nano C100 setup

- The official [OKDO Nano C100 System Image](https://auto.designspark.info/okdo_images/c100.img.xz) is based on Ubuntu 18.04 LTS (Bionic Beaver)
- The default Jetpack version is 4.6.1, but might have to be installed manually with ```sudo apt install -y nvidia-jetpack```
- The default python version is 3.6

### Installation of tensorflow with cuda support on OKDO Nano C100

> [!IMPORTANT] 
> A common issue that occurs when instsalling tensorflow is ```fatal error: xlocale.h: No such file or directory```. This can be fixed by creating the symlink ```sudo ln -s /usr/include/locale.h /usr/include/xlocale.h```

The Python version that should be used is 3.6, and the tensorflow version that should be used on the Jetson Nano is tensorflow-2.7.0+nv22. The general recommended installation procedure is as follows:

```bash
sudo apt install -y python3-dev pkg-config
sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo python -m pip3 install --verbose 'protobuf<4' 'Cython<3'

sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
```

Now install the tensorflow package with the following command:

```bash
sudo python -m pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
```

## Desktop/Server setup

The project can be run on a desktop/server environment. The project has been tested on Ubuntu 20.04 LTS. The following distro dependencies are required:

- Ubuntu 20.04 LTS
- Python >=3.8
- CUDA >=10
- cuDNN >=8.0.5
- Tensorflow >=2.7.0

It is generally recommended to handle the cuda dependencies using conda. ```scripts/init.sh``` will install the necessary dependencies for the project.