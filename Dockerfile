FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN conda install -c conda-forge gdal
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install rasterio
RUN pip install numpy
RUN pip install wandb
RUN pip install tmuxp
RUN pip install tabulate
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tmux git -y