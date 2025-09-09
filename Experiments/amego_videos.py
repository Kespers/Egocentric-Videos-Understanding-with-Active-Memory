#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea un video annotato da JSON + frame + tabella oggetti.

Uso:
python script.py --video_id 46
"""

import argparse, json, hashlib, traceback
from pathlib import Path
from collections import defaultdict

import torch
from torchvision.io import read_image
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

# ----------------- CONFIG -----------------
BASE_PATH = Path("/workspace/amego/enigma-51/AMEGO")
OBJECTS_SUBFOLDER = "HOI_AMEGO"
OUTPUT_VIDEOS = BASE_PATH / "ENIGMA-AMB/OUTPUT_VIDEOS"
BOX_WIDTH = 2
HEADER_HEIGHT = 40
TABLE_ROW_HEIGHT = 30
TABLE_ROWS = 5  # numero di righe visibili
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Larghezze colonne
COL_WIDTHS = [150, 150, 100, 180, 200]  # cluster, track_id, color, time_left_active, time_to_be_active
TOTAL_TABLE_WIDTH = sum(COL_WIDTHS) + 20  # padding

# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ----------------- UTILS -----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def try_int(x):
    if x is None: return None
    if isinstance(x,int): return x
    if isinstance(x,float): return int(round(x))
    if isinstance(x,str):
        try: return int(round(float(x.strip())))
        except: return None
    return None

def normalize_frames(num_frame_field):
    frames = []
    if num_frame_field is None: return frames
    if isinstance(num_frame_field,(list,tuple)):
        for el in num_frame_field:
            i = try_int(el)
            if i is not None: frames.append(i)
    else:
        i = try_int(num_frame_field)
        if i is not None: frames.append(i)
    return sorted(set(frames))

def normalize_bboxes(obj_bbox_field):
    bboxes=[]
    if obj_bbox_field is None: return bboxes
    def parse_one(b):
        if not isinstance(b,(list,tuple)) or len(b)!=4: return None
        try: return (float(b[0]),float(b[1]),float(b[2]),float(b[3]))
        except: return None
    if isinstance(obj_bbox_field,(list,tuple)) and len(obj_bbox_field)>0:
        first = obj_bbox_field[0]
        if isinstance(first,(int,float,str)):
            p = parse_one(obj_bbox_field)
            if p: bboxes.append(p)
        else:
            for item in obj_bbox_field:
                p = parse_one(item)
                if p: bboxes.append(p)
    else:
        p = parse_one(obj_bbox_field)
        if p: bboxes.append(p)
    return bboxes

def deterministic_color_for_id(id_value):
    key = str(id_value).encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    def boost(x): return int(50 + (x/255)*(230-50))
    return (boost(r), boost(g), boost(b))


# ----------------- LOAD ANNOTATIONS -----------------
def load_annotations(json_path):
    mapping = defaultdict(list)
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    for i,elem in enumerate(data):
        try:
            track_id = elem.get("track_id",None)
            cluster = elem.get("cluster", "?")
            frames_raw = elem.get("num_frame")
            bboxes_raw = elem.get("obj_bbox")
            if isinstance(frames_raw,(list,tuple)) and isinstance(bboxes_raw,(list,tuple)) and len(frames_raw)==len(bboxes_raw):
                for fn_raw,bb_raw in zip(frames_raw,bboxes_raw):
                    fn = try_int(fn_raw)
                    bb_list = normalize_bboxes(bb_raw)
                    if fn is not None and bb_list:
                        mapping[fn].append({
                            "track_id":track_id or f"unk_{i}",
                            "cluster": cluster,
                            "bbox":bb_list[0]
                        })
                continue
            frames = normalize_frames(frames_raw or elem.get("last_frame") or elem.get("frame"))
            bboxes = normalize_bboxes(bboxes_raw)
            if not frames or not bboxes: continue
            for fn in frames:
                for bb in bboxes:
                    mapping[fn].append({
                        "track_id":track_id or f"unk_{i}",
                        "cluster": cluster,
                        "bbox":bb
                    })
        except Exception:
            print(f"Warning: errore lettura elemento {i}")
            traceback.print_exc()
            continue
    return mapping


# ----------------- DRAW -----------------
def draw_boxes_and_labels(img_pil, boxes):
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    for b in boxes:
        x,y,w,h = b['bbox']
        color = deterministic_color_for_id(b['track_id'])
        x1,y1,x2,y2 = int(x),int(y),int(x+w),int(y+h)

        draw.rectangle([x1,y1,x2,y2], outline=color, width=BOX_WIDTH)
        text = f"{b['track_id']} | {b['cluster']}"
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.rectangle([x1,y1-th, x1+tw+4, y1], fill="white")
        draw.text((x1+2,y1-th), text, fill="black", font=font)
    return img_pil


def build_track_history(frame_to_boxes):
    history = defaultdict(lambda: {"cluster":"?", "frames":[]})
    for fn, boxes in frame_to_boxes.items():
        for b in boxes:
            tid = b["track_id"]
            history[tid]["cluster"] = b["cluster"]
            history[tid]["frames"].append(fn)
    for tid in history:
        history[tid]["frames"] = sorted(set(history[tid]["frames"]))
    return dict(sorted(history.items(), key=lambda x: str(x[0])))


def add_header(img_pil, video_id, idx, total_frames, canvas_w):
    frame_w, frame_h = img_pil.size
    header = Image.new("RGB",(canvas_w,HEADER_HEIGHT),"white")
    draw = ImageDraw.Draw(header)
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except:
        font = ImageFont.load_default()
    text = f"video_id: {video_id}   frame {idx}/{total_frames}"
    # Calcola dimensione testo con textbbox
    bbox = draw.textbbox((0,0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((canvas_w-tw)//2, 5), text, fill="black", font=font)

    new_img = Image.new("RGB",(canvas_w, frame_h + HEADER_HEIGHT))
    x_offset = (canvas_w-frame_w)//2
    new_img.paste(header,(0,0))
    new_img.paste(img_pil,(x_offset,HEADER_HEIGHT))
    return new_img


def add_table(img_pil, track_history, current_frame, canvas_w):
    frame_w, frame_h = img_pil.size
    table_h = TABLE_ROW_HEIGHT*(TABLE_ROWS+1)
    table = Image.new("RGB",(canvas_w,table_h),"white")
    draw = ImageDraw.Draw(table)
    try:
        font = ImageFont.truetype(FONT_PATH, 18)
    except:
        font = ImageFont.load_default()

    headers = ["cluster", "track_id", "color", "time_left_active", "time_to_be_active"]
    total_col_width = sum(COL_WIDTHS)
    x_start = (canvas_w - total_col_width) // 2
    x_positions = [x_start]
    for w in COL_WIDTHS[:-1]:
        x_positions.append(x_positions[-1]+w)

    y0 = 10
    for i,hdr in enumerate(headers):
        draw.text((x_positions[i], y0), hdr, fill="black", font=font)

    rows = []
    for tid,data in track_history.items():
        frames = data["frames"]
        cluster = data["cluster"]
        color = deterministic_color_for_id(tid)
        if current_frame in frames:
            time_left = max(frames)-current_frame+1
            time_to_be = 0
        else:
            future = [f for f in frames if f>current_frame]
            time_left = 0
            time_to_be = future[0]-current_frame if future else -1

        rows.append({
            "tid":tid,
            "cluster":cluster,
            "color":color,
            "time_left":time_left,
            "time_to_be":time_to_be,
            "active": current_frame in frames
        })

    rows.sort(key=lambda r: (not r["active"], r["time_to_be"] if r["time_to_be"]>=0 else 1e9))
    y = y0 + 30
    for row in rows[:TABLE_ROWS]:
        draw.text((x_positions[0], y), str(row["cluster"]), fill="black", font=font)
        draw.text((x_positions[1], y), str(row["tid"]), fill="black", font=font)
        draw.rectangle([x_positions[2], y, x_positions[2]+25, y+20], fill=row["color"], outline="black")
        draw.text((x_positions[3], y), str(row["time_left"]), fill="black", font=font)
        draw.text((x_positions[4], y), str(row["time_to_be"]), fill="black", font=font)
        y += TABLE_ROW_HEIGHT

    new_img = Image.new("RGB",(canvas_w,frame_h+table_h))
    new_img.paste(img_pil,(0,0))
    new_img.paste(table,(0,frame_h))
    return new_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=int, required=True)
    args = parser.parse_args()
    video_id = args.video_id

    frames_folder = BASE_PATH / f"{video_id}/rgb_frames"
    json_file = BASE_PATH / OBJECTS_SUBFOLDER / f"{video_id}.json"
    output_video_file = OUTPUT_VIDEOS / f"annotated_{video_id}.mp4"
    ensure_dir(OUTPUT_VIDEOS)

    files = sorted([p for p in frames_folder.iterdir() if p.is_file() and p.suffix.lower()==".jpg"])
    if not files:
        print("Nessun frame trovato.")
        return

    frame_to_boxes = load_annotations(json_file)
    track_history = build_track_history(frame_to_boxes)

    first_frame = cv2.imread(str(files[0]))
    frame_h, frame_w = first_frame.shape[:2]
    fps = 30
    total_h = frame_h + HEADER_HEIGHT + TABLE_ROW_HEIGHT*(TABLE_ROWS+1)
    total_w = max(frame_w, TOTAL_TABLE_WIDTH)
    out_vid = cv2.VideoWriter(str(output_video_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))

    print(f"Creazione video {output_video_file} con {len(files)} frame...")

    for idx,f in enumerate(files,1):
        frame_num = int(f.stem.split('_')[-1])
        boxes = frame_to_boxes.get(frame_num,[])

        img_tensor = read_image(str(f)).to(device)
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray(img_np)

        img_pil = draw_boxes_and_labels(img_pil, boxes)
        img_pil = add_header(img_pil, video_id, idx, len(files), total_w)
        img_pil = add_table(img_pil, track_history, frame_num, total_w)

        img_bgr_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        out_vid.write(img_bgr_final)
        print(f"\rFrame {idx}/{len(files)}", end="", flush=True)

    out_vid.release()
    print(f"\nVideo salvato in: {output_video_file}")


if __name__=="__main__":
    main()