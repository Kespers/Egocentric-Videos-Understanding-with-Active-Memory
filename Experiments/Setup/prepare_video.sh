#!/bin/bash
set -euo pipefail

# Attiva conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Controllo argomenti
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_root> <video_id>"
    exit 1
fi

# Parametri
base_root="$1"
video_id="$2"
root="${base_root}/${video_id}"
video_path="${root}.mp4"
fps=30


echo "[STARTED][ID=${video_id}]"

echo "[(1/5) - FRAMES][ID=${video_id}]"
frames_dir="${root}/rgb_frames"
mkdir -p "$frames_dir"
ffmpeg -i "${video_path}" -vf "fps=${fps},scale=456:256" "${frames_dir}/frame_%010d.jpg"

# Step 2: flowformer + HOI extractor in parallelo
echo "[(2/5) - FLOWFORMER][ID=${video_id}]"
conda activate amego
python -m tools.generate_flowformer_flow --root ${root} --v_id ${video_id} --dset video --models_root submodules/flowformer/models --model sintel --video_fps ${fps}

echo "[(3/5) - HOI][ID=${video_id}]"
hoi_dir="${root}/hand-objects"
conda activate handobj
mkdir -p ${hoi_dir}

python -m tools.extract_bboxes --image_dir ${frames_dir} --cuda --mGPUs --checksession 1 --checkepoch 8 --checkpoint 132028 --bs 32 --detections_pb ${hoi_dir}/${video_id}.pb2
python -m submodules.epic-kitchens-100-hand-object-bboxes.src.scripts.convert_raw_to_releasable_detections ${hoi_dir}/${video_id}.pb2 ${hoi_dir}/${video_id}.pkl --frame-height 256 --frame-width 456

echo "[(4/5) - HOI AMEGO][ID=${video_id}]"
python HOI_AMEGO.py --dset video --v_id "${video_id}" --video_fps "${fps}"

echo "[(5/5) - LS AMEGO][ID=${video_id}]"
python LS_AMEGO.py --dset video --v_id "${video_id}" --video_fps "${fps}"


echo "[FINISH][ID=${video_id}]"