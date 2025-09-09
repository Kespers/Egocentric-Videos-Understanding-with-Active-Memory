import os
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def load_json(video_id, mem_type):
    MEM_TYPE = "HOI_AMEGO" if mem_type == "HOI" else "LS_AMEGO"
    json_path = f"/workspace/amego/enigma-51/AMEGO/{MEM_TYPE}/{video_id}.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def draw_and_save_frame(frame_id, bbox, frames_dir, out_dir, video_id):
    frame_path = os.path.join(frames_dir, f"frame_{frame_id:010d}.jpg")
    if not os.path.exists(frame_path):
        return f"[WARN] Missing frame: {frame_path}"

    try:
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        x, y, w, h = map(lambda v: int(round(v)), bbox)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=4)

        out_path = os.path.join(out_dir, f"{video_id}_{frame_id:010d}.jpg")
        img.save(out_path)
        return None
    except Exception as e:
        return f"[ERROR] {frame_path}: {e}"


def group_by_cluster(hoi, mem_type):
    clusters = {}
    for item in hoi:
        cluster_id = item["cluster"]

        if cluster_id not in clusters:
            clusters[cluster_id] = {}

        if mem_type == "LS":
            # LS → un solo "track" fittizio per cluster
            track_id = "ls_track"
            if track_id not in clusters[cluster_id]:
                clusters[cluster_id][track_id] = {"num_frame": [], "obj_bbox": []}

            clusters[cluster_id][track_id]["num_frame"].extend(item["num_frame"])
            clusters[cluster_id][track_id]["obj_bbox"].extend(
                [(0, 0, 1, 1)] * len(item["num_frame"])
            )
        else:
            # HOI → usa track_id reali
            track_id = item["track_id"]
            num_frames = item["num_frame"]
            obj_bboxes = item["obj_bbox"]

            if track_id not in clusters[cluster_id]:
                clusters[cluster_id][track_id] = {"num_frame": [], "obj_bbox": []}

            clusters[cluster_id][track_id]["num_frame"].extend(num_frames)
            clusters[cluster_id][track_id]["obj_bbox"].extend(obj_bboxes)

    return clusters


def process_cluster(video_id, mem_type, clusters, output_root, workers=8):
    frames_dir = f"/workspace/amego/enigma-51/AMEGO/{video_id}/rgb_frames"
    tasks = []

    for cluster_id, tracks in clusters.items():
        for track_id, track_data in tracks.items():
            # LS → salva direttamente nella cartella del cluster
            if mem_type == "LS":
                out_dir = os.path.join(output_root, str(cluster_id))
            else:
                out_dir = os.path.join(output_root, str(cluster_id), str(track_id))

            os.makedirs(out_dir, exist_ok=True)

            for frame_id, bbox in zip(track_data["num_frame"], track_data["obj_bbox"]):
                tasks.append((frame_id, bbox, frames_dir, out_dir, video_id))

    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(draw_and_save_frame, *task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            err = f.result()
            if err:
                errors.append(err)

    if errors:
        print("\n".join(errors))


def main(video_id, mem_type, workers=8):
    final_path = "HOI_FRAMES" if mem_type == "HOI" else "LS_FRAMES"
    output_root = f"/workspace/amego/enigma-51/AMEGO/ENIGMA-AMB/{final_path}"

    hoi = load_json(video_id, mem_type)
    clusters = group_by_cluster(hoi, mem_type)
    process_cluster(video_id, mem_type, clusters, output_root, workers=workers)

    print(f"[DONE] Processed video_id={video_id}. Output in {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=int, required=True)
    parser.add_argument("--mem_type", type=str, required=True, choices=["HOI", "LS"])
    parser.add_argument("--workers", type=int, default=8, help="Numero di processi paralleli")
    args = parser.parse_args()

    main(args.video_id, args.mem_type, workers=args.workers)