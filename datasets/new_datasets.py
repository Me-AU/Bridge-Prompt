import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class GTEA_KPD(data.Dataset):
    def __init__(self,
                 root='./data/gtea',
                 transform=None, mode='val',
                 num_frames=16, ds=None, ol=None,
                 small_test=False,
                 frame_dir="./data/gtea/png/",
                 label_dir="./data/gtea/xml_labels/",
                 pretrain=True,
                 n_split=None):
        if ds is None:
            ds = [1, 2, 4]
        if ol is None:
            ol = [1, 1, 0.5]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.ds = ds
        self.overlap = ol
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.pretrain = pretrain
        self.n_split = n_split

        self.annotations = self.load_annotations()

    def load_annotations(self):
        annotations = []
        for xml_file in os.listdir(self.label_dir):
            if xml_file.endswith('.xml'):
                tree = ET.parse(os.path.join(self.label_dir, xml_file))
                root = tree.getroot()
                for image in root.findall('image'):
                    frame_id = image.get('id')
                    keypoints = []
                    for polyline in image.findall('polyline'):
                        points = polyline.get('points')
                        keypoints.append([float(p) for p in points.split(',')])
                    annotations.append((frame_id, keypoints))
        return annotations

    def frame_sampler(self, videoname, vlen):
        start_idx = int(videoname[1])
        ds = videoname[2]
        seq_idx = np.arange(self.num_frames) * int(ds) + start_idx
        seq_idx = np.where(seq_idx < vlen, seq_idx, vlen - 1)
        return seq_idx

    def __getitem__(self, index):
        frame_id, keypoints = self.annotations[index]
        vpath = os.path.join(self.frame_dir, frame_id)
        vlen = len([f for f in os.listdir(vpath) if os.path.isfile(os.path.join(vpath, f))])
        path_list = os.listdir(vpath)
        path_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        frame_index = self.frame_sampler((frame_id, 0, 1), vlen)
        seq = [Image.open(os.path.join(vpath, path_list[i])).convert('RGB') for i in frame_index]

        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return seq, keypoints

    def __len__(self):
        return len(self.annotations)


class GTEA_FRAMES(data.Dataset):
    def __init__(self,
                 root='./data/gtea',
                 small_test=False,
                 frame_dir='./data/gtea/frames/',
                 save_feat_dir='gtea_vit_features',
                 transform=None):
        self.root = root
        self.small_test = small_test
        self.frame_dir = frame_dir
        self.save_feat_dir = save_feat_dir
        self.transform = transform

        all_files = os.walk(self.frame_dir)
        self.convert_tensor = transforms.ToTensor()
        self.data_lst = []
        for path, dir, filelst in all_files:
            if len(filelst) > 0:
                self.data_lst.append((filelst, path))

    def __getitem__(self, index):
        videoname = self.data_lst[index]
        vroot = videoname[1]
        path_list = videoname[0]
        # vlen = len(path_list)
        path_list.sort(key=lambda x: int(x[4:-4]))
        seq = [Image.open(os.path.join(vroot, p)).convert('RGB') for p in path_list]
        if self.transform is not None:
            seq = self.transform(seq)
        else:
            convert_tensor = transforms.ToTensor()
            seq = [convert_tensor(img) for img in seq]
            seq = torch.stack(seq)
        vsplt = vroot.split('/')[-1]
        fname = vsplt + '.npy'
        return seq, fname

    def __len__(self):
        return len(self.data_lst)


if __name__ == '__main__':
    dataset = GTEA_KPD()
    dataloader = data.DataLoader(dataset, batch_size=12, shuffle=True)
    for i, (img, keypoints) in enumerate(dataloader):
        print(img, keypoints)
