from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import torchaudio

class Sound8KDataset(Dataset):
    def __init__(self, paths, info_df, transform = None):
        self.transform = transform
        self.info_df = info_df
        self.file_list = []
        for path in paths:
            self.file_list.extend(glob("%s/**/*.wav"%path, recursive=True))
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        filepath = self.file_list[index]
        _, filename = os.path.split(filepath)
        label = int(self.info_df[self.info_df['slice_file_name']==filename]['classID'])
        waveform, sample_rate = torchaudio.load(filepath)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

class ReadData():
