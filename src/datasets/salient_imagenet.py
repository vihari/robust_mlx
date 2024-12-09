import pandas as pd
import numpy as np
import os
from collections import defaultdict

from PIL import Image

import torch as ch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch

from src.datasets.standard_dataset import DatasetWithSpecifications

from . import wordnet_dict

IMAGENET_PATH = '/scratch/ead54/data/imagenet/'  #'/home/data/ImageNet/'
SALIENT_IMAGENET_PATH = '/scratch/vp421/salient_imagenet_dataset' # '/home/data/salient_imagenet_dataset/'
CSV_PATH = '/scratch/vp421/salient_imagenet_dataset/discover_spurious_features.csv' #'/mnt/ssd/jj/j_Research/IBP_based_XAI_AP_training/src/datasets/discover_spurious_features.csv'


def get_simagenet(class_idxs, core=False):
    """
    :param core: when set to true, core feature indices instead of spurious are used
    """
    dats = []
    spu_dict = MTurk_Results(CSV_PATH).spurious_features_dict
    core_dict = MTurk_Results(CSV_PATH).core_features_dict
    
    # wordnet_ids = [wordnet_dict.mapping[cidx] for cidx in class_idxs]
    for class_index in class_idxs:
        spu_fidxs = spu_dict[class_index]
        core_fidxs = core_dict[class_index]
        print(spu_fidxs, core_fidxs, class_index)
        if not core:
            dats.append(SalientImageNet(IMAGENET_PATH, SALIENT_IMAGENET_PATH, class_index, spu_fidxs))
        else:
            dats.append(SalientImageNet(IMAGENET_PATH, SALIENT_IMAGENET_PATH, class_index, core_fidxs))
        
    final_dat = MyConcatDataset(dats)
    return final_dat
    
    
class MyConcatDataset(DatasetWithSpecifications):
    def __init__(self, dats):
        self.dats = dats
        self.full_dat = ConcatDataset(self.dats)
        self.labels = []
        for i, _dat in enumerate(self.dats):
            self.labels += [i]*len(_dat)

        rng = np.random.default_rng(42)
        _ln = len(self.labels)
        all_idxs = rng.permuted(list(range(_ln)))
        train_ln, val_ln, test_ln = int(_ln*0.6), int(_ln*0.15), int(_ln*0.25)
        train_idxs, val_idxs, test_idxs = all_idxs[:train_ln], all_idxs[train_ln:train_ln+val_ln], all_idxs[train_ln+val_ln:]
        self.split_dict = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}
    
    def __getitem__(self, idx):
        img, mask = self.full_dat[idx]
        # dummy group label
        return img, self.labels[idx], mask, 0
    
    def __len__(self):
        return len(self.labels)
    
    def targets(self):
        return np.array(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.dats)
        

class SalientImageNet(Dataset):
    def __init__(self, images_path, masks_path, class_index, feature_indices, 
                 resize_size=256, crop_size=224):
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

        wordnet_dict = eval(open(os.path.join(masks_path, 'wordnet_dict.py')).read())
        wordnet_id = wordnet_dict[class_index]
        
        self.images_path = os.path.join(images_path, 'train', wordnet_id)
        self.masks_path = os.path.join(masks_path, wordnet_id)
        
        image_names_file = os.path.join(self.masks_path, 'image_names_map.csv')
        image_names_df = pd.read_csv(image_names_file)
        
        image_names = []
        feature_indices_dict = defaultdict(list)
        for feature_index in feature_indices:
            image_names_feature = image_names_df[str(feature_index)].to_numpy()
            
            for i, image_name in enumerate(image_names_feature):
                image_names.append(image_name)                
                feature_indices_dict[image_name].append(feature_index)        
        
        self.image_names = np.unique(np.array(image_names))                
        self.feature_indices_dict = feature_indices_dict
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = os.path.join(self.images_path, image_name + '.JPEG')

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)
        image = image.resize((self.crop_size, self.crop_size))
        
        feature_indices = self.feature_indices_dict[image_name]
        
        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in feature_indices:            
            curr_mask_path = os.path.join(self.masks_path, 'feature_' + str(feature_index), image_name + '.JPEG')
            
            mask = np.asarray(Image.open(curr_mask_path))
            mask = (mask/255.)
            
            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        mask_tensor = torch.permute(mask_tensor, [1, 2, 0])
        # print(mask_tensor.shape)
        return np.array(image), np.array(mask_tensor)
    



##############################################################################################################
# Utils

import numpy as np
import pandas as pd
from collections import defaultdict

def print_with_stars(print_str, total_count=115, prefix="", suffix="", star='*'):
    str_len = len(print_str)
    left_len = (total_count - str_len)//2
    right_len = total_count - left_len - str_len
    final_str = "".join([star]*(left_len)) + print_str + "".join([star]*(right_len))
    final_str = prefix + final_str + suffix
    print(final_str)
    
def topk_predictive_features(class_index, robust_model, robust_features, k=5):
    W = (robust_model.model.fc.weight).detach().cpu().numpy()
    W_class = W[class_index: class_index + 1, :]
    FI_values = np.mean(robust_features * W_class, axis=0)

    features_indices = np.argsort(-FI_values)[:k]
    return features_indices

class MTurk_Results:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)
        
        self.aggregate_results(self.dataframe)
        print(f"Read {csv_path} and read {len(self.answers_dict)} answers")
        self.class_feature_maps()
        self.core_spurious_labels_dict()
        self.spurious_feature_lists()

        
    def aggregate_results(self, dataframe):
        answers_dict = defaultdict(list)
        reasons_dict = defaultdict(list)
        feature_rank_dict = defaultdict(int)
        wordnet_dict = defaultdict(str)
        
        for row in dataframe.iterrows():
            index, content = row
            WorkerId = content['WorkerId']

            class_index = int(content['Input.class_index'])
            feature_index = int(content['Input.feature_index'])
            feature_rank = int(content['Input.feature_rank'])
            
            wordnet_dict[class_index] = content['Input.wordnet_id']
            
            key = str(class_index) + '_' + str(feature_index)

            main_answer = content['Answer.main']
            confidence = content['Answer.confidence']
            reasons = content['Answer.reasons']
            
            answers_dict[key].append((WorkerId, main_answer, confidence, reasons))
            reasons_dict[key].append(reasons)
                
            feature_rank_dict[key] = feature_rank

        self.answers_dict = answers_dict
        self.feature_rank_dict = feature_rank_dict
        self.reasons_dict = reasons_dict
        self.wordnet_dict = wordnet_dict
    
    def core_spurious_labels_dict(self):
        answers_dict = self.answers_dict
        
        core_features_dict = defaultdict(list)
        spurious_features_dict = defaultdict(list)
        
        core_spurious_dict = {}
        core_list = []
        spurious_list = []
        for key, answers in answers_dict.items():
            class_index, feature_index = key.split('_')
            class_index, feature_index = int(class_index), int(feature_index)
            
            num_spurious = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer in ['separate_object', 'background']:
                    num_spurious = num_spurious + 1                

            if num_spurious >= 3:
                spurious_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'spurious'
                spurious_list.append(key)
                
            else:
                core_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'core'
                core_list.append(key)
                
        self.core_spurious_dict = core_spurious_dict
        self.core_list = core_list
        self.spurious_list = spurious_list
        
        self.core_features_dict = core_features_dict
        self.spurious_features_dict = spurious_features_dict
    
    def spurious_feature_lists(self):
        answers_dict = self.answers_dict
        
        background_list = []
        separate_list = []
        ambiguous_list = []
        for key, answers in answers_dict.items():
            num_background = 0
            num_separate = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer == 'background':
                    num_background = num_background + 1
                elif main_answer == 'separate_object':
                    num_separate = num_separate + 1
                                
            if num_background >= 3:
                background_list.append(key)
            elif num_separate >= 3:
                separate_list.append(key)
            elif (num_background + num_separate) >= 3:
                ambiguous_list.append(key)
                
        self.background_list = background_list
        self.separate_list = separate_list
        self.ambiguous_list = ambiguous_list
        
        
    def class_feature_maps(self):
        answers_dict = self.answers_dict
        
        keys_list = answers_dict.keys()
        
        feature_to_classes_dict = defaultdict(list)
        class_to_features_dict = defaultdict(list)
        
        for key in keys_list:
            class_index, feature_index = key.split('_')
            class_index = int(class_index)
            feature_index = int(feature_index)
            
            feature_to_classes_dict[feature_index].append(class_index)
            class_to_features_dict[class_index].append(feature_index)
            
        self.class_to_features_dict = class_to_features_dict
        self.feature_to_classes_dict = feature_to_classes_dict
        