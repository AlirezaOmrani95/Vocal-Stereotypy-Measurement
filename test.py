import torch
from torch import nn
import torchaudio as ta
from torchaudio import transforms as tt
from utils import _mix_down_if_necessary_, _resample_if_necessary_
from tqdm import tqdm
import numpy as np
import os
from pretrained_models import Pretrained_Models
import random
import argparse

def arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default= './dataset/test/', help= 'The address to read data from')
    parser.add_argument('--target_sample_rate',default= 22050 ,help= 'The location of input CSV file')
    parser.add_argument('--best_weight_dir', default= './weights/best weight/best_weight', help= 'The location of the Best weight file')
    parser.add_argument('--threshold', default= 0.5, help= 'The threshold for the classifier')
    return(parser.parse_args())

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model,data,device):
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
        pred = torch.sigmoid(logits)

    return pred

if __name__ == '__main__':
    
    #Dataset Info
    args = arg_parsing()
    files_list = os.listdir(args.dataset_dir)
    for file in files_list:
        print(f'\n{file}:')
        audio_file,sample_rate = ta.load(os.path.join(args.dataset_dir,file))
        audio_file_seconds = []

        #Feature extractor
        mel_spectogram_transform = tt.MelSpectrogram(args.target_sample_rate,n_fft= 1024 , hop_length=256,
                                                    n_mels=128)
        mfcc_transform = tt.MFCC(args.target_sample_rate,128,melkwargs={'n_fft': 1024,
                                                        "n_mels": 128,
                                                        "hop_length":256,
                                                        'mel_scale':"htk"})
        if audio_file.shape[0]>1:
            audio_file = _mix_down_if_necessary_(audio_file)
        if sample_rate != args.target_sample_rate:
            audio_file, sample_rate = _resample_if_necessary_(audio_file,sample_rate,args.target_sample_rate)
        for i in range(int(audio_file.shape[1]/sample_rate)):
            if (i+1) <= int(audio_file.shape[1]/sample_rate):
                sample = audio_file[:,i*sample_rate:(i+1)*sample_rate]
                sample_mel_spectogram = mel_spectogram_transform(sample)
                sample__mfcc = mfcc_transform(sample)
                audio_file_seconds.append(torch.concat([sample_mel_spectogram,sample__mfcc]))

        #Implementing the model
        model = Pretrained_Models('xcit_tiny_12_p8_224',10,(2,128,345)).get_model()
        model = model.to(device)

        model.head = nn.Linear(model.head.in_features,1).to(device)
        model.load_state_dict(torch.load(args.best_weight_dir)['model_state_dict'])
        audio_file_seconds = torch.from_numpy(np.array(audio_file_seconds))   
        results = np.array(test(model, audio_file_seconds, device).cpu())
        results = (results > args.threshold).astype(int)
        print(f'''{np.count_nonzero(results==1)} / {len(results)} seconds of the total audio was predicted as vocal stereotypy.\nIn another word, {np.count_nonzero(results==1) / len(results)*100 :.2f} percentage of the file has been recognized as vocal stereotypy''')
