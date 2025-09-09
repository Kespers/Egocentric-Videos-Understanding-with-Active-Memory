#!/bin/bash
set -e  # esce se un comando fallisce

echo "Creazione ambiente amego..."
source activate base
conda activate amego
pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
cd submodules/epic-kitchens-100-hand-object-bboxes
python setup.py install
cd ../../
conda deactivate
echo "Creazione ambiente handobj..."
conda activate handobj
cd submodules/hand_object_detector/
pip install -r requirements.txt -q
cd lib
export CUDA_HOME=$CONDA_PREFIX/pkgs/cuda-toolkit
conda install cudatoolkit-dev=11.3 -c pytorch -c nvidia -c conda-forge
python setup.py build develop
cd ../../
pip install protobuf==3.20.3 -q
pip install imageio -q
conda deactivate
echo "Setup completato!"
