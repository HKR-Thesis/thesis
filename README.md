# About the Project

This project is a simple example of training a pendulum to balance using Q learning. The pendulum is a simple inverted pendulum system with a single joint and a mass at the end. The pendulum is trained using a Q learning algorithm to balance the pendulum in the upright position.


https://github.com/HKR-Thesis/traininig_environment_pendulum/assets/89925489/8556d586-fc47-44ae-9b20-d89ca00a4ce2

# OKDO Nano C100 setup

## Notes

- The default Jetpack version is 4.6.1
- A common issue that occurs when instsalling tensorflow is ```fatal error: xlocale.h: No such file or directory```. This can be fixed by running ```sudo ln -s /usr/include/locale.h /usr/include/xlocale.h```
- The default python version is 3.6

## Installation of tensorflow with cuda support on OKDO Nano C100

The Python version that should be used is 3.6. This way we can either make a docker image or install the packages directly on the device. The prerequisites are the following:

```bash
sudo apt install -y python3-dev pkg-config
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install --verbose 'protobuf<4' 'Cython<3'
sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
```

Now install the tensorflow package with the following command:

```bash
python3.6 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.7.0+nv22.1
```