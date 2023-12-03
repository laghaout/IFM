FROM python:3.10-slim

# Declare the working directory.
WORKDIR /home

# Install Linux utilities.
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install less
RUN apt -y install tk
RUN apt -y install tree
RUN apt -y install git

# Install Python packages.
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install seaborn==0.12.2
RUN pip3 install matplotlib==3.6.2
RUN pip3 install numpy==1.23.5
RUN pip3 install pandas==1.5.2
RUN pip3 install pytest
RUN pip3 install py-cpuinfo
#RUN pip3 install sequana --upgrade
RUN pip3 install dcor==0.6
RUN pip3 install pre-commit
RUN pip3 install poetry

# Temporarily downgrade `protobuf` to avoid <https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly>
RUN pip3 install protobuf==3.20.*

# Install the local package.
COPY . .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "ifm/main.py"]
