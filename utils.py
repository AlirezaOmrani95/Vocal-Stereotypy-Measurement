import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio as ta
from torchaudio import transforms as tt
import numpy as np
import random as rnd
import os
import pandas as pd

rnd.seed(1)
np.random.seed(1)

class Audio_Dataset(Dataset):
    def __init__(self, root,annotations, device='cpu'):
        super().__init__()
        self.device = device
        self.annotations = annotations
        self.root = root

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        file_name = f'{self.annotations.iloc[index,0]}_{self.annotations.iloc[index,1].strip()}_{self.annotations.iloc[index,2].strip()}.npy'.replace('_aug','')
        folder_name = self.annotations.iloc[index,0]
        label = self._get_audio_sample_label_(index)

        if label == 2:
            label = 1

        mel_spectogram_audio = torch.from_numpy(np.load(os.path.join(self.root,folder_name,'MelSpectogram',
                                                    file_name.replace(':','-'))))
        mfcc_audio_audio = torch.from_numpy(np.load(os.path.join(self.root,folder_name,'MFCC',
                                                file_name.replace(':','-'))))
        input_ = torch.concat([mel_spectogram_audio,mfcc_audio_audio],dim=0)

        return input_, label
    def _get_class_num_ (self):
        classes = np.zeros(2)
        for idx in range(self.__len__()):
            if self._get_audio_sample_label_(idx) == 0:
                classes[0] +=1
            else:
                classes[1] +=1

        return classes
    def _get_audio_sample_label_(self,index):
        return self.annotations.iloc[index,3]

def _resample_if_necessary_(audio_file, original_sample_rate, target_sample_rate):
        resampler = tt.Resample(original_sample_rate,target_sample_rate)
        return resampler(audio_file), target_sample_rate

def _mix_down_if_necessary_(audio_file):
    #reducing the number of channels into 1
    return torch.mean(audio_file,dim= 0,keepdim=True)

def train_valid_separation(annotation_dir, valid_rate=0.1):
    valid_indices = []
    annotations = pd.read_csv(annotation_dir)
    ones = annotations.index[annotations.iloc[:,-1] != 0].tolist()
    zeros = annotations.index[annotations.iloc[:,-1] == 0].tolist()
    
    valid_indices.extend(np.random.choice(ones,int(np.ceil(len(ones)*valid_rate))))
    valid_indices.extend(np.random.choice(zeros,int(np.ceil(len(zeros)*valid_rate))))

    train_indices = list(set(list(range(len(annotations)))) - set(valid_indices))

    valid_annotations = annotations.loc[valid_indices,:]
    train_annotations = annotations.loc[train_indices,:]
    return train_annotations,valid_annotations

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch':epoch},save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def create_data_loader(dataset, batch_size):
    return DataLoader(dataset,batch_size,shuffle=True)

def get_session_indices(data,session_list):
        session_annotation = {}
        
        for session in session_list:
            row_indices = data.index[data[2] == session].tolist()
            session_annotation[session] = str(row_indices[0])+'-'+str(row_indices[-1])
        return session_annotation