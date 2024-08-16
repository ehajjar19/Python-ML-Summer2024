# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the environment.yaml file into the container
COPY environment.yaml .

# Create the Conda environment
RUN conda env create -f environment.yaml

# Activate the environment and ensure it's the default environment
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]

#Copy Files into the container
COPY rsfmri_dti-main /usr/src/app

# Run directory
RUN