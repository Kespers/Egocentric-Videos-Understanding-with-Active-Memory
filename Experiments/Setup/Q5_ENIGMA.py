import argparse
from .EnigmaLoader import EnigmaDataset

import pandas as pd
import torch
import os, json
import sys
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
from tools.data import EPICDataset


import os
from PIL import Image


def find_cooccurrences(summary, cluster_key):
    clusters = summary[cluster_key].unique()
    co_occurrences = {}
    for cluster in clusters:
        co_occurrences[cluster] = set()
        same_cluster = summary[summary[cluster_key] == cluster]
        for i, row in same_cluster.iterrows():
            current_start = row['num_frame'][0]
            current_stop = row['num_frame'][-1]
            for j, row2 in summary.iterrows():
                if row2[cluster_key] == row[cluster_key]:
                    continue
                other_start = row2['num_frame'][0]
                other_stop = row2['num_frame'][-1]
                if (current_start <= other_start <= current_stop) or (other_start <= current_start <= other_stop):
                    co_occurrences[cluster].add(row2[cluster_key])
        co_occurrences[cluster] = list(co_occurrences[cluster])
    return co_occurrences
    
def extract_object_features(dset, crops, video):
    all_features = []
    for crop in crops:
        frame, bbox = crop[0], crop[1]
        image = dset.load_image((video, frame))
        width, height = image.size
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        cropped_image = image.crop(bbox)
        cropped_image = dset.transform(cropped_image)
        with torch.no_grad():
            features = net(cropped_image.unsqueeze(0).cuda())
            all_features.append(features.detach().cpu())
            
    stacked_features = torch.stack(all_features, dim=0)
    average_features = torch.mean(stacked_features, dim=0, keepdim=True).squeeze(1)
    return average_features

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root', type=str, required=True, help='EPIC root folder')
parser.add_argument('--AMEGO', type=str, required=True, help='AMEGO folder')
parser.add_argument('--dataset_type', type=str, required=True, help='Video id')
parser.add_argument('--fps', type=int, required=True, help='Video fps')

args = parser.parse_args()

dataset_file = f'AMB/Q5_ENIGMA_TEST_{args.dataset_type}.json'
print("USING", dataset_file)
Q5 = pd.read_json(dataset_file)
dset = EnigmaDataset(args.root, args.fps)
correctly_answered = 0
old_video = None

features_key = 'features'
cluster_key = 'cluster'
net = torch.hub.load('facebookresearch/dinov2:qasfb-patch-1', 'dinov2_vitl14')
net = nn.DataParallel(net)
net.eval().cuda()
results = {}
for _, question in tqdm(Q5.iterrows(), total=len(Q5)):
    video_id = question['video_id']
    
    if video_id != old_video:
        old_video = video_id
        summary = pd.read_json(os.path.join(args.AMEGO, 'HOI_AMEGO', f'{video_id}.json'))
        co_occurrences = find_cooccurrences(summary, cluster_key)    
        summary_avg_features = torch.stack([torch.tensor(feat) for feat in summary[features_key].values], dim=0).squeeze(1)
    
    correct = question['correct']
    obj = list(question['question_image'].keys())[0]
    question_image = question['question_image'][obj]
    predicted_seq = {}
    average_features = extract_object_features(dset, question_image, video_id)
    cos_sim = F.cosine_similarity(average_features, summary_avg_features)
    object_found = int(torch.argmax(cos_sim))
    object_found = summary.iloc[object_found][cluster_key]
    obj_cooc = co_occurrences[object_found]   
    
    score = {}
    for answer, answer_images in question['answers'].items():
        summary_cooc = 0
        for i, answer_image in answer_images.items():
            average_features = extract_object_features(dset, answer_image, video_id)
            cos_sim = F.cosine_similarity(average_features, summary_avg_features)
            object_found = int(torch.argmax(cos_sim))
                    
            object_found = summary.iloc[object_found][cluster_key]
            if object_found in obj_cooc:
                summary_cooc += 1
        score[answer] = summary_cooc / len(answer_images)
    
    maximum = max(score.values())
    answer_given = random.choice([k for k, v in score.items() if v == maximum])
    
    results[question['id']] = [answer_given, correct]

    if int(answer_given) == int(correct):
        correctly_answered += 1

    print(
        f"[{question['id']}]: [{correct} - {answer_given}] "
        f"[CORRECT:{correctly_answered}/{len(Q5)}",
        flush=True
    )
        
    
print(f"Average: {correctly_answered}/{len(Q5)}")    

percentuale = correctly_answered / len(Q5) * 100
obj = {
    "dataset_type": args.dataset_type,
    "results": f"{correctly_answered}/{len(Q5)}",
    "percentuale": f"{percentuale:.2f}%",
    "response": results
}

os.makedirs("Results_AMEGO", exist_ok=True)
results_path = f"Results_AMEGO/RESULTS_{args.dataset_type}.json"

# carica se già esiste
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
else:
    data = []

data.append(obj)

with open(results_path, "w") as f:
    json.dump(data, f, indent=2)

print("File aggiornato correttamente.")