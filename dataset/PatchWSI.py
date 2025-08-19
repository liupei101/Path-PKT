"""
Class for bag-style dataloader
"""
from typing import Union
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.io import retrieve_from_table_clf
from utils.io import read_patch_data, read_patch_coord
from utils.func import sampling_data, random_mask_instance
from utils.func import agg_dict, fill_placeholder
from .label_converter import MetaSurvData


class WSIPatchSurv(Dataset):
    r"""A WSI dataset class for survival prediction tasks (patient-level generally).

    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        mode (string): 'patch', or 'cluster'.
        meta_data: label information of all samples in the dataset.
        read_format (string): The suffix name or format of the file storing patch feature.
    Return:
        index: The index of current item in the whole dataset, used to retrieval patient ID.
        (feats, extra_data): Patch features and extra data.
        label: It contains typical survival labels, 'last follow-up time' and 'event status';
            event = 1 -> w/ event, called uncensored one; event = 0 -> w/o event, called censored one.
    """
    def __init__(self, patient_ids: list, patch_path: str, mode:str, meta_data:Union[list,MetaSurvData],
        read_format:str='pt', sampling_ratio:Union[None,float,int]=None, sampling_seed=42, **kws):
        super().__init__()
        if sampling_ratio is not None:
            print("[dataset] Patient-level sampling with ratio ({}) and seed ({})".format(sampling_ratio, sampling_seed))
            patient_ids, pid_left = sampling_data(patient_ids, sampling_ratio, seed=sampling_seed)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        assert mode in ['patch', 'cluster']
        self.mode = mode
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        self.kws = kws
        
        self.pids, self.pid2info = meta_data.collect_info_by_pids(
            patient_ids, target_columns=['pathology_id', 'y_t', 'y_e', 'project', 'dataset', 'dataset_id']
        )

        self.meta_data = meta_data
        self.uid = self.pids
        self.read_path = patch_path
        self.read_format = read_format

        self.summary()

    def summary(self):
        print(f"[Dataset] WSIPatchSurv: in {self.mode} mode, avaiable patients count {self.__len__()}.")

    def get_meta_data(self):
        return self.meta_data

    def get_patient_info(self):
        return self.pids, self.pid2info

    def get_feat_read_path(self, pid, sid):
        if "{project}" in self.read_path:
            cur_read_path = self.read_path.replace("{project}", self.pid2info[pid]['project'])
        return osp.join(cur_read_path, sid + '.' + self.read_format)

    def read_patch_data_from_patient(self, pid, sids):
        feats = []
        for sid in sids:
            full_path = self.get_feat_read_path(pid, sid)
            if not osp.exists(full_path):
                raise ValueError(f"[WSIPatchSurv] not found slide {sid} in {full_path}.")
            slide_feat = read_patch_data(full_path, dtype='torch')
            if len(slide_feat.shape) == 3 and slide_feat.shape[0] == 1:
                slide_feat = slide_feat.squeeze(0)
            feats.append(slide_feat)

        feats = torch.cat(feats, dim=0).to(torch.float)
        return feats

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        info  = self.pid2info[pid]
        sids  = info['pathology_id']
        label = [info['y_t'], info['y_e']]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = self.read_patch_data_from_patient(pid, sids)
            extra_data = torch.Tensor([info['dataset_id']]).int() # 0 if there is only one dataset

            return index, (feats, extra_data), label

        elif self.mode == 'cluster':
            cids = np.load(osp.join(self.kws['cluster_path'], '{}.npy'.format(pid)))
            feats = []
            for sid in sids:
                full_path = self.get_feat_read_path(pid, sid)
                if not osp.exists(full_path):
                    raise ValueError(f"[WSIPatchSurv] not found slide {sid} in {full_path}.")
                feats.append(read_patch_data(full_path, dtype='torch'))
            feats = torch.cat(feats, dim=0).to(torch.float)
            cids = torch.Tensor(cids)
            assert cids.shape[0] == feats.shape[0]
            return index, (feats, cids), label

        else:
            pass
            return None

 
class WSIPatchSurv_Transfer(WSIPatchSurv):
    r"""A WSI dataset class for survival prediction tasks (patient-level generally).
    Different from `WSIPatchSurv`, this class supports loading transfer data.

    Args:
        transfer_feat_path: the path to transfer features to be loaded, e.g.,
        /path/to/transfer-feats/target_tcga_stes/source_tcga_ucec-fold_0/
    
    Please refer to `WSIPatchSurv` for more details.
    """
    def __init__(self, patient_ids: list, patch_path: str, transfer_feat_path: str, mode:str, 
        meta_data:Union[list,MetaSurvData], read_format:str='pt', sampling_ratio:Union[None,float,int]=None, 
        sampling_seed=42, sel_feat_idx=None, self_transfer_feat_path=None, **kws):
        super().__init__(patient_ids, patch_path, mode, meta_data, read_format, sampling_ratio, sampling_seed, **kws)
        
        # setup the path to transfer data
        self.transfer_feat_path = transfer_feat_path
        print(f"[WSIPatchSurv_Transfer] Transfer data will be loaded from {transfer_feat_path}.")

        # if use original patch features
        if 'with_patch_feat' in kws and kws['with_patch_feat']:
            self.load_patch_feat = True
            print(f"[WSIPatchSurv_Transfer] Original patch features will be loaded.")
        else:
            self.load_patch_feat = False
            print(f"[WSIPatchSurv_Transfer] Original patch features will not be loaded.")

        # use the transfer features specified by `sel_feat_idx`
        if sel_feat_idx is not None:
            self.sel_feat_idx = sel_feat_idx
            print(f"[WSIPatchSurv_Transfer] these transfer features will be used: {sel_feat_idx}.")
        else:
            self.sel_feat_idx = None

        if self_transfer_feat_path is not None:
            self.self_transfer_feat_path = self_transfer_feat_path
            print(f"[WSIPatchSurv_Transfer] note that self-transfer features will be used: {self_transfer_feat_path}.")
        else:
            self.self_transfer_feat_path = None

    def get_transfer_feat_read_path(self, pid):
        return osp.join(self.transfer_feat_path, pid + '.' + self.read_format)

    def get_self_transfer_feat_read_path(self, pid):
        return osp.join(self.self_transfer_feat_path, pid + '.' + self.read_format)

    def __getitem__(self, index):
        pid   = self.pids[index]
        info  = self.pid2info[pid]
        sids  = info['pathology_id']
        label = [info['y_t'], info['y_e']]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            full_transfer_path = self.get_transfer_feat_read_path(pid)
            transfer_feat = read_patch_data(full_transfer_path, dtype='torch')
            if self.sel_feat_idx is not None:
                transfer_feat = transfer_feat[self.sel_feat_idx]

            if self.self_transfer_feat_path is not None:
                self_transfer_path = self.get_self_transfer_feat_read_path(pid)
                self_transfer_feat = read_patch_data(self_transfer_path, dtype='torch')
                if len(self_transfer_feat.shape) == 1:
                    self_transfer_feat = self_transfer_feat.unsqueeze(0) # [1, d]
                transfer_feat = torch.cat([self_transfer_feat, transfer_feat], dim=0)

            # original patch features as `extra_data`
            if self.load_patch_feat:
                feats = self.read_patch_data_from_patient(pid, sids)
            else:
                feats = torch.Tensor([0])

            return index, (transfer_feat, feats), label

        else:
            pass
            return None


class WSIPatchClf(Dataset):
    r"""A WSI dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, label_path:Union[None,str]=None,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, ratio_mask=None, mode='patch', **kws):
        super(WSIPatchClf, self).__init__()
        if ratio_sampling is not None:
            assert ratio_sampling > 0 and ratio_sampling < 1.0
            print("[dataset] patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))
        if ratio_mask is not None and ratio_mask > 1e-5:
            assert ratio_mask <= 1, 'The argument ratio_mask must be not greater than 1.'
            assert mode == 'patch', 'Only support a patch mode for instance masking.'
            self.ratio_mask = ratio_mask
            print("[dataset] masking instances with ratio_mask = {}".format(ratio_mask))
        else:
            self.ratio_mask = None

        self.read_path = patch_path
        self.label_path = label_path
        self.has_patch_label = (label_path is not None) and len(label_path) > 0
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')
        self.uid = self.sids
        
        assert mode in ['patch', 'cluster']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        self.summary()

    def summary(self):
        print(f"[dataset] in {self.mode} mode, avaiable WSIs count {self.__len__()}")
        if not self.has_patch_label:
            print("[dataset] the patch-level label is not avaiable, derived by slide label.")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        if self.mode == 'patch':
            full_path = osp.join(self.read_path, sid + '.' + self.read_format)
            feats = read_patch_data(full_path, dtype='torch').to(torch.float)
            # if masking patches
            if self.ratio_mask:
                feats = random_mask_instance(feats, self.ratio_mask, scale=1, mask_way='mask_zero')
            full_coord = osp.join(self.kws['coord_path'],  sid + '.h5')
            coors = read_patch_coord(full_coord, dtype='torch')
            if self.has_patch_label:
                path = osp.join(self.label_path, sid + '.npy')
                patch_label = read_patch_data(path, dtype='torch', key='label').to(torch.long)
            else:
                patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
            assert patch_label.shape[0] == feats.shape[0]
            assert coors.shape[0] == feats.shape[0]
            return index, (feats, coors), (label, patch_label)
        else:
            pass
            return None