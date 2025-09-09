import pandas as pd
import os
from PIL import Image
from tools.transforms import default_transform
from tools.data import get_video_metadata

class EnigmaDataset:
    def __init__(self, root, fps):
        self.root = root
        self.transform = default_transform('val')
        self.name = 'video'

        self.frame_shape = {}
        self.video_fps = {}
        self.video_length = {}

        # trova tutte le sottocartelle con i video (nomi numerici)
        self.v_ids = sorted([
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d)) and d.isdigit()
        ])

        for v_id in self.v_ids:
            rgb_path = os.path.join(self.root, f'{v_id}/rgb_frames')
            frame_shape, video_length = get_video_metadata(rgb_path)

            self.frame_shape[v_id] = frame_shape
            self.video_fps[v_id] = fps
            self.video_length[v_id] = video_length

            print(f"[{v_id}] Frame shape: {frame_shape}, FPS: {fps}, Video length: {video_length} frames")

    def frame_path(self, img):
        v_id, f_id = img
        return os.path.join(self.root, f'{v_id}/rgb_frames/frame_{f_id:010d}.jpg')

    def flowformer_path(self, img):
        v_id, f_id = img
        return os.path.join(self.root, f'{v_id}/flowformer/flow_{f_id:010d}.pth')

    def load_image(self, img):
        file = self.frame_path(img)
        return Image.open(file).convert('RGB')

    def frames_root(self, v_id):
        return os.path.join(self.root, f'{v_id}/rgb_frames')
    
    def flowformer_root(self, v_id):
        return os.path.join(self.root, f'{v_id}/flowformer')
        
    def detections_path(self, v_id):
        return os.path.join(self.root, f'{v_id}/hand-objects/{v_id}.pkl')