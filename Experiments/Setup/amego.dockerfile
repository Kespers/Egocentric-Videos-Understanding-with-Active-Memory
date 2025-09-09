FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Installa pacchetti di sistema necessari
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ wget git ffmpeg ca-certificates bzip2 \
        libglib2.0-0 libxext6 libsm6 libxrender1 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 2. Installa Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 3. Aggiungi Miniconda al PATH
ENV PATH="/opt/conda/bin:$PATH"

# 4. Accetta termini di servizio (necessario per Conda >=23)
RUN conda init && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 5. Imposta directory di lavoro
WORKDIR /workspace/amego

COPY amego.yml .
RUN conda env create -f amego.yml -y && conda create --name handobj python=3.8 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# 10. Copia tutto il resto del progetto
COPY . .

RUN chmod +x KS_setup/extract_flowformer.sh
RUN chmod +x KS_setup/extract_frames.sh
RUN chmod +x KS_setup/extract_HOI.sh
RUN chmod +x KS_setup/prepare_video_OPT.sh
RUN chmod +x KS_setup/setup_enigma_amego.sh

RUN chmod +x KS_setup/setup_amego_env.sh && bash KS_setup/setup_amego_env.sh

# 11. Avvia una shell bash quando il container parte
ENTRYPOINT ["bash"]
