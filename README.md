# About

Our project investigates the potential of Q-Learning, a reinforcement learning technique, for controlling an inverted pendulum (cart-pole) system in resource-constrained
embedded environments, compared to traditional machine learning computing setups. Highlighting the importance of efficient resource management, we aim to explore how Q-Learning can manage complex tasks with minimal resources while maintaining accuracy, using the cart-pole—a benchmark experiment for non-linear control—as our testbed. Our study is motivated by the transformative impact of reinforcement learning on autonomous systems, especially in navigation and adaptation within dynamic environments.

We evaluate the performance and resource consumption of Q-Learning on an efficient embedded platform, such as the NVIDIA Jetson Nano, against more resource-abundant computing environments. Our research focuses on assessing the trade-offs in performance metrics like response time, accuracy, and energy usage, aiming to establish whether Q-Learning can achieve comparable effectiveness in embedded systems as it does in larger computing environments. By demonstrating the practicality of Q-Learning for real-time or on-device training in embedded systems, our work seeks to advance autonomous system development for scenarios where computing resources are limited. This could lead to more energy-efficient, cost-effective autonomous solutions across various applications, addressing the challenges of speed, efficiency, and power consumption critical to edgecomputing devices

# Installation

The project provides several setup scripts under scripts/ to install the necessary dependencies. An assumption before initializing or running for the Jetson Nano is that tensorflow has already been setup on the device (instructions can be found [below](#installation-of-tensorflow-with-cuda-support-on-okdo-nano-c100)). The scripts are as follows:

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

The Python version that should be used is 3.6, and the tensorflow version that should be used on the [OKDO Nano](https://www.okdo.com/p/okdo-nano-c100-developer-kit-powered-by-nvidia-jetson-nano-module/) is tensorflow-2.7.0+nv22. The [official installation procedure](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770) is as follows:

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
