FROM amazonlinux:2
RUN mkdir -p /usr/src/
WORKDIR /usr/src/
COPY requirements-local.txt /usr/src/

RUN amazon-linux-extras install python3.8 -y
# インストール済みのパッケージを最新版にアップデート
RUN yum -y update && \
    yum -y install \
    sudo \
    zip \
    unzip \
    which \
    awscil\
    less\
    glibc\
    groff\
    python38-devel

RUN echo 'alias python=python3.8' >> ~/.bashrc
RUN source ~/.bashrc

RUN pip3.8 install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements-local.txt

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.1.30.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install
RUN rm awscliv2.zip
RUN rm -r aws
