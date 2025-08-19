from .func import parse_str_dims, fill_placeholder

# strictly ordered, DO NOT CHANGE THIS
DATASET_LIST = [
    'tcga_blca', 'tcga_brca', 'tcga_cesc', 'tcga_coadread', 'tcga_gbmlgg', 
    'tcga_hnsc', 'tcga_kipan', 'tcga_lihc', 'tcga_lung', 'tcga_sarc', 
    'tcga_skcm', 'tcga_stes', 'tcga_ucec'
]
MAP_TO_NEAREST_TRANSFER = {
    'tcga_blca': 'tcga_hnsc',
    'tcga_brca': 'tcga_gbmlgg',
    'tcga_cesc': 'tcga_stes',
    'tcga_coadread': 'tcga_stes',
    'tcga_gbmlgg': 'tcga_kipan',
    'tcga_hnsc': 'tcga_blca',
    'tcga_kipan': 'tcga_gbmlgg',
    'tcga_lihc': 'tcga_sarc',
    'tcga_lung': 'tcga_kipan',
    'tcga_sarc': 'tcga_lihc',
    'tcga_skcm': 'tcga_stes',
    'tcga_stes': 'tcga_cesc',
    'tcga_ucec': 'tcga_sarc',
}

MAP_TO_MIDDLE_TRANSFER = {
    'tcga_blca': 'tcga_gbmlgg',
    'tcga_brca': 'tcga_coadread',
    'tcga_cesc': 'tcga_coadread',
    'tcga_coadread': 'tcga_brca',
    'tcga_gbmlgg': 'tcga_lihc',
    'tcga_hnsc': 'tcga_gbmlgg',
    'tcga_kipan': 'tcga_blca',
    'tcga_lihc': 'tcga_skcm',
    'tcga_lung': 'tcga_stes',
    'tcga_sarc': 'tcga_kipan',
    'tcga_skcm': 'tcga_lihc',
    'tcga_stes': 'tcga_brca',
    'tcga_ucec': 'tcga_coadread',
}

MAP_TO_FAREST_TRANSFER = {
    'tcga_blca': 'tcga_brca',
    'tcga_brca': 'tcga_skcm',
    'tcga_cesc': 'tcga_kipan',
    'tcga_coadread': 'tcga_gbmlgg',
    'tcga_gbmlgg': 'tcga_stes',
    'tcga_hnsc': 'tcga_sarc',
    'tcga_kipan': 'tcga_skcm',
    'tcga_lihc': 'tcga_stes',
    'tcga_lung': 'tcga_skcm',
    'tcga_sarc': 'tcga_skcm',
    'tcga_skcm': 'tcga_gbmlgg',
    'tcga_stes': 'tcga_sarc',
    'tcga_ucec': 'tcga_blca',
}

MAP_TO_POS_TRANSFER_IDXS = {
    'tcga_blca': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'tcga_brca': [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
    'tcga_cesc': [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    'tcga_coadread': [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12],
    'tcga_gbmlgg': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
    'tcga_hnsc': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
    'tcga_kipan': [1, 4, 7, 8],
    'tcga_lihc': [1, 2, 3, 6, 9],
    'tcga_lung': [1, 2, 3, 4, 5, 6, 9, 11, 12],
    'tcga_sarc': [7],
    'tcga_skcm': [0, 5, 8, 11],
    'tcga_stes':  [0, 1, 2, 3, 5, 7, 8, 10, 12],
    'tcga_ucec': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

def convert_to_abbr(key):
    ABBR_MAPS = {
        'data_split_fold': 'fold',
        'dataset_name': 'data',
        'dataset_chosen_index': 'data',
        'test_source_dataset_name': 'src',
        'test_source_data_split_fold': 'src_fold',
        'transfer_feat_type': 'tsrc',
        'transfer_source_dataset': 'tsrc',
        'transfer_source_fold': 'tsrc_fold',
        'auxtfl_aux_branch': 'mff',
        'auxtfl_scale': 'scale',
        'auxtfl_fusion': 'fuse',
        'decoder_num_feat_proj_layers': 'num_prj',
        'deepmil_post_mil_layer': 'post',
        'moetfl_expert_network': 'enet',
        'moetfl_expert_topk': 'topk',
        'loss_balance_weight': 'w_ba',
        'loss_router_z_weight': 'w_rz',
        'mfftfl_mff': 'mff',
    }

    if key in ABBR_MAPS.keys():
        print(f"[info] abbreviate {key} as {ABBR_MAPS[key]}.")
        return ABBR_MAPS[key]
    else:
        return key

def ignore_it_in_save_path(key, value):
    IGNORE_LIST = dict()

    if key in IGNORE_LIST.keys():
        judge_func = IGNORE_LIST[key]
        return judge_func(value)

    return False

def get_exp_datasets(cfg):
    """
    Return a list of datasets that will be used in this experiment.
    """
    if 'dataset_name' not in cfg or cfg['dataset_name'] is None:
        # Multiple datasets for this experiment
        assert 'dataset_list' in cfg and 'dataset_chosen_index' in cfg, "Failed to find datasets for experiments."
        dataset_list = parse_str_dims(cfg['dataset_list'], dtype=str)
        dataset_chosen_index = parse_str_dims(cfg['dataset_chosen_index'], dtype=int)
        dataset_chosen_index = sorted(dataset_chosen_index)
        return [dataset_list[i] for i in dataset_chosen_index]
    else:
        # One dataset for this experiment
        assert isinstance(cfg['dataset_name'], str)
        return [cfg['dataset_name']]

def fill_placeholder_in_cfg(cfg):
    # for placeholder = {dataset}
    if 'dataset_name' in cfg:
        dataset_name = cfg['dataset_name']
        temp_keys = [
            'save_path', 'path_patch', 'path_coord', 'path_cluster', 'path_graph', 
            'path_table', 'data_split_path', 'test_save_wsi_representation_path', 
            'test_load_ckpt_path', 'transfer_path_feat', 'transfer_load_ckpt_path', 
            'transfer_path_self_feat',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], dataset_name, ind="{dataset}")

    # for placeholder = {fold}
    if 'data_split_fold' in cfg and cfg['data_split_fold'] is not None:
        data_split_fold = cfg['data_split_fold']
        temp_keys = [
            'data_split_path', 'path_table_data_split', 'transfer_path_feat', 
            'transfer_load_ckpt_path', 'transfer_path_self_feat',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], data_split_fold, ind="{fold}")

    # add a new key `transfer_source_dataset` for source_dataset
    if 'transfer_learning' in cfg and cfg['transfer_learning']:
        if 'transfer_feat_type' in cfg and cfg['transfer_feat_type'] in ['near_tf', 'mid_tf', 'far_tf', 'self_tf']:
            if cfg['transfer_feat_type'] == 'near_tf':
                cfg['transfer_source_dataset'] = MAP_TO_NEAREST_TRANSFER[cfg['dataset_name']]
            elif cfg['transfer_feat_type'] == 'mid_tf':
                cfg['transfer_source_dataset'] = MAP_TO_MIDDLE_TRANSFER[cfg['dataset_name']]
            elif cfg['transfer_feat_type'] == 'far_tf':
                cfg['transfer_source_dataset'] = MAP_TO_FAREST_TRANSFER[cfg['dataset_name']]
            elif cfg['transfer_feat_type'] == 'self_tf':
                cfg['transfer_source_dataset'] = cfg['dataset_name']
                cfg['transfer_source_fold'] = cfg['data_split_fold']
        elif 'transfer_source_dataset' not in cfg or cfg['transfer_source_dataset'] is None:
            cfg['transfer_source_dataset'] = 'all'
        print(f"[INFO] setup `transfer_source_dataset` = {cfg['transfer_source_dataset']}.")

    # transfer learing: for placeholder = {source_dataset}
    if 'transfer_source_dataset' in cfg and 'transfer_learning' in cfg and cfg['transfer_learning']:
        source_dataset_name = cfg['transfer_source_dataset']
        temp_keys = [
            'transfer_path_feat', 'transfer_load_ckpt_path',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], source_dataset_name, ind="{source_dataset}")

    # transfer learing: for placeholder = {source_fold}
    if 'transfer_source_fold' in cfg and 'transfer_learning' in cfg and cfg['transfer_learning']:
        source_fold = cfg['transfer_source_fold']
        temp_keys = [
            'transfer_path_feat', 'transfer_load_ckpt_path',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], source_fold, ind="{source_fold}")

    # zero transfer test: for placeholder = {source_dataset}
    if 'test_source_dataset_name' in cfg and cfg['test']:
        source_dataset_name = cfg['test_source_dataset_name']
        temp_keys = [
            'test_load_ckpt_path', 'test_save_wsi_representation_path',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], source_dataset_name, ind="{source_dataset}")

    # zero transfer test: for placeholder = {source_fold}
    if 'test_source_data_split_fold' in cfg and cfg['test']:
        source_fold = cfg['test_source_data_split_fold']
        temp_keys = [
            'test_load_ckpt_path', 'test_save_wsi_representation_path',
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], source_fold, ind="{source_fold}")

    # add new keys
    if 'transfer_learning' in cfg and cfg['transfer_learning'] and 'transfer_feat_type' in cfg:
        if cfg['transfer_feat_type'] == 'all_tf':
            sel_idx = []
            for i in range(len(DATASET_LIST)):
                if DATASET_LIST[i] == cfg['dataset_name']:
                    continue
                sel_idx.append(i)
        elif cfg['transfer_feat_type'] == 'pos_tf':
            # the followings are the zero-transfers that lead to positive C-Index (> 0.5)
            sel_idx = MAP_TO_POS_TRANSFER_IDXS[cfg['dataset_name']]
        else:
            sel_idx = None
        cfg['transfer_feat_idx'] = sel_idx

        # if need to fill `moetfl_expert_size`
        if 'moetfl_expert_size' in cfg:
            num_experts = len(sel_idx) if sel_idx is not None else 0
            num_experts += 1 if 'transfer_self_feat' in cfg and cfg['transfer_self_feat'] else 0
            if num_experts != cfg['moetfl_expert_size']:
                print(f"[INFO] found {num_experts} transfer models used; moetfl_expert_size is changed from {cfg['moetfl_expert_size']} to {num_experts}.")
                cfg['moetfl_expert_size'] = num_experts

        # if need to fill `moetfl_expert_topk`
        if 'moetfl_expert_topk' in cfg:
            if cfg['moetfl_expert_topk'] is None or cfg['moetfl_expert_topk'] < 0:
                expert_topk = 1 + len(MAP_TO_POS_TRANSFER_IDXS[cfg['dataset_name']])
                print(f"[INFO] found moetfl_expert_topk = {cfg['moetfl_expert_topk']}; changed to {expert_topk} (num_pos_tf).")
                cfg['moetfl_expert_topk'] = expert_topk
        
    return cfg

def check_necessary_columns_in_label_dataframe(columns):
    for col in ['patient_id', 'pathology_id', 'project', 'dataset', 'dataset_id', 't', 'e']:
        assert col in columns, f"Column named {col} is not found in label dataframe."

def is_valid_run_cfg(cfg):
    if 'test' in cfg and cfg['test']:
        if cfg['test_source_dataset_name'] == cfg['dataset_name'] and cfg['data_split_fold'] != cfg['test_source_data_split_fold']:
            print("[INFO] found the same dataset name but different fold for the source and the target task (at test mode).")
            return False

    if 'transfer_learning' in cfg and cfg['transfer_learning']:
        if 'transfer_source_dataset' in cfg and cfg['transfer_source_dataset'] == cfg['dataset_name'] and cfg['data_split_fold'] != cfg['transfer_source_fold']:
            print("[INFO] found the same dataset name but different fold for the source and the target task (at transfer learning mode).")
            return False

    return True