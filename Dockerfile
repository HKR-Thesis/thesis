FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

WORKDIR /app
COPY . /app

RUN apt install llvm-9 llvm-10

RUN echo "export LLVM_CONFIG=/usr/bin/llvm-config-9" >> ~/.bashrc

RUN cd /usr/bin && ln -sf python3.6 python
RUN python -m pip install --upgrade pip && python -m pip install -r jetson-requirements.txt

EXPOSE 80

ENTRYPOINT ["bash"]

