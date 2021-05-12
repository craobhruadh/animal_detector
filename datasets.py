import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

# for reference:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


class cats_dogs_dataset(Dataset):

    @staticmethod
    def prep_annotations(directory='../data/train/'):
        df = pd.DataFrame()
        df['filename'] = os.listdir(directory)
        df['class'] = df['filename'].apply(lambda x: 'cat' if 'cat' in x
                                           else 'dog')
        df.to_csv(os.path.join(directory, '../data/anotations.csv'))

    def __init__(self,
                 annotations='../data/anotations.csv',
                 img_dir='../data/train/',
                 transform=None):
        self.img_labels = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir)
        image = read_image(path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.img_labels)


if __name__ == "__main__":
    training_data = cats_dogs_dataset()
