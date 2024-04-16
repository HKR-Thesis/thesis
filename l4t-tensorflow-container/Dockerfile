FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y openssh-server llvm-9 llvm-10

ENV LLVM_CONFIG=/usr/bin/llvm-config-9

RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo "export VISIBLE=now" >> /etc/profile

RUN cd /usr/bin && ln -sf python3.6 python
RUN python -m pip install --upgrade pip && python -m pip install -r jetson-requirements.txt

EXPOSE 22 80

CMD ["/usr/sbin/sshd", "-D"]

