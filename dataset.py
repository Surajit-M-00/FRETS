import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from scipy.signal import butter, filtfilt
import cv2

from torchvision import transforms
from PIL import Image




# 3. Data Synchronization and DataLoader
def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def resample(input_signal, target_length):
    input_signal = np.asarray(input_signal)
    return np.asarray(np.interp(np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal))

def extract_synchronized_spo2(json_path, number_of_frame):
    with open(json_path, 'r') as f:
        data = json.load(f)
    spo2_values = [item['Value']['o2saturation'] for item in data['/FullPackage']]
    return resample(spo2_values, number_of_frame)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def extract_dc_ac_components(spatio_temporal_map, fs=30):
    b_lp, a_lp = butter_lowpass(cutoff=0.3, fs=fs, order=5)
    dc_component = filtfilt(b_lp, a_lp, spatio_temporal_map, axis=2)
    b_bp, a_bp = butter_bandpass(lowcut=0.75, highcut=2.5, fs=fs, order=5)
    ac_component = filtfilt(b_bp, a_bp, spatio_temporal_map, axis=2)
    return dc_component, ac_component



# 4. Chunk-based DataLoader for training
class train_dataset(Dataset):
    def __init__(self, map_files, map_dir, chunk_size=300):
        self.map_files = map_files
        self.map_dir = map_dir
        self.chunk_size = chunk_size
        
        # Precompute all chunks for efficient access
        self.chunk_data = []
        for map_file in self.map_files:
            file = np.load(os.path.join(self.map_dir, map_file))
            spatio_temporal_map = file['video']
            number_of_frame = spatio_temporal_map.shape[2]
            synchronized_spo2 = file['wave']
            fps = file['fps']
            
            for start_frame in range(0, number_of_frame - chunk_size + 1, chunk_size):
                end_frame = start_frame + chunk_size
                spatio_temporal_map_chunk = spatio_temporal_map[:, :, start_frame:end_frame]
                dc_component, ac_component = extract_dc_ac_components(spatio_temporal_map_chunk, fs=fps)
                dc_tensor = torch.tensor(dc_component.copy(), dtype=torch.float32)
                ac_tensor = torch.tensor(ac_component.copy(), dtype=torch.float32)
                st_tensor = torch.tensor(spatio_temporal_map_chunk.copy(), dtype=torch.float32) 
                spo2_tensor = torch.tensor(synchronized_spo2[start_frame:end_frame], dtype=torch.float32)

                # Clean the NaNs
                dc_tensor = torch.nan_to_num(dc_tensor, nan=0.0)
                ac_tensor = torch.nan_to_num(ac_tensor, nan=0.0)
                st_tensor = torch.nan_to_num(st_tensor, nan=0.0)
                spo2_tensor = torch.nan_to_num(spo2_tensor, nan=0.0)


                self.chunk_data.append((st_tensor,dc_tensor, ac_tensor, spo2_tensor))

    def __len__(self):
        return len(self.chunk_data)
    
    def __getitem__(self, idx):
        return self.chunk_data[idx]


# class train_dataset(Dataset):
#     def __init__(self, map_files, map_dir, chunk_size=300):
#         self.map_files = map_files
#         self.map_dir = map_dir
#         self.chunk_size = chunk_size

#         self.transform = transforms.Compose([
#                     transforms.ToTensor(),             # Convert the image to a PyTorch tensor
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
#                                         std=[0.229, 0.224, 0.225])
#                 ])
        
#         # Precompute all chunks for efficient access
#         self.chunk_data = []
#         for map_file in self.map_files:
#             file = np.load(os.path.join(self.map_dir, map_file))
#             spatio_temporal_map = file['video']
#             number_of_frame = spatio_temporal_map.shape[2]
#             synchronized_spo2 = file['wave']
            
#             for start_frame in range(0, number_of_frame - chunk_size + 1, chunk_size):
#                 end_frame = start_frame + chunk_size
#                 spatio_temporal_map_chunk = spatio_temporal_map[0:3, :, start_frame:end_frame]
#                 dc_component, ac_component = extract_dc_ac_components(spatio_temporal_map_chunk)


#                 st_tensor = self.transform(np.transpose(spatio_temporal_map_chunk.copy(), (1, 2, 0))).type(torch.float32)
#                 ac_tensor = self.transform(np.transpose(ac_component.copy(), (1, 2, 0))).type(torch.float32)
#                 dc_tensor = self.transform(np.transpose(dc_component.copy(), (1, 2, 0))).type(torch.float32)

#                 st_tensor = st_tensor[[2, 1, 0], :, :]
#                 ac_tensor = ac_tensor[[2, 1, 0], :, :]
#                 dc_tensor = dc_tensor[[2, 1, 0], :, :]


#                 spo2_tensor = torch.tensor(synchronized_spo2[start_frame:end_frame], dtype=torch.float32)
#                 self.chunk_data.append((st_tensor,dc_tensor, ac_tensor, spo2_tensor))

#     def __len__(self):
#         return len(self.chunk_data)
    
#     def __getitem__(self, idx):
#         return self.chunk_data[idx]

class test_dataset(Dataset):
    def __init__(self, map_files, map_dir):
        self.map_files = map_files
        self.map_dir = map_dir

    def __len__(self):
        return len(self.map_files)
    
    def __getitem__(self, idx):
        map_file = self.map_files[idx]
        file = np.load(os.path.join(self.map_dir, map_file))
        spatio_temporal_map = file['video']
        synchronized_spo2 = file['wave']
        fps = file['fps']
        dc_component, ac_component = extract_dc_ac_components(spatio_temporal_map[:,:,:],fs=fps)
        dc_tensor = torch.tensor(dc_component.copy(), dtype=torch.float32)
        ac_tensor = torch.tensor(ac_component.copy(), dtype=torch.float32)
        spo2_tensor = torch.tensor(synchronized_spo2, dtype=torch.float32)
        st_tensor = torch.tensor(spatio_temporal_map[:,:,:].copy(), dtype=torch.float32)


        dc_tensor = torch.nan_to_num(dc_tensor, nan=0.0)
        ac_tensor = torch.nan_to_num(ac_tensor, nan=0.0)
        st_tensor = torch.nan_to_num(st_tensor, nan=0.0)
        spo2_tensor = torch.nan_to_num(spo2_tensor, nan=0.0)


        return st_tensor, dc_tensor, ac_tensor, spo2_tensor  


# class test_dataset(Dataset):
#     def __init__(self, map_files, map_dir):
#         self.map_files = map_files
#         self.map_dir = map_dir
#         self.transform = transforms.Compose([
#                     transforms.ToTensor(),             # Convert the image to a PyTorch tensor
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
#                                         std=[0.229, 0.224, 0.225])
#                 ])

#     def __len__(self):
#         return len(self.map_files)
    
#     def __getitem__(self, idx):
#         map_file = self.map_files[idx]
#         file = np.load(os.path.join(self.map_dir, map_file))
#         spatio_temporal_map = file['video']
#         synchronized_spo2 = file['wave']
#         spatio_temporal_map = spatio_temporal_map[0:3,:,:]
#         dc_component, ac_component = extract_dc_ac_components(spatio_temporal_map)


#         st_tensor = self.transform(np.transpose(spatio_temporal_map.copy(), (1, 2, 0))).type(torch.float32)
#         ac_tensor = self.transform(np.transpose(ac_component.copy(), (1, 2, 0))).type(torch.float32)
#         dc_tensor = self.transform(np.transpose(dc_component.copy(), (1, 2, 0))).type(torch.float32)

#         st_tensor = st_tensor[[2, 1, 0], :, :]
#         ac_tensor = ac_tensor[[2, 1, 0], :, :]
#         dc_tensor = dc_tensor[[2, 1, 0], :, :]

        

#         spo2_tensor = torch.tensor(synchronized_spo2, dtype=torch.float32)
#         return st_tensor, dc_tensor, ac_tensor, spo2_tensor    
