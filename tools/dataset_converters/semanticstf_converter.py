import os
from os import path as osp
from pathlib import Path

import mmengine

cor_list = [
    'train',
    'snow',
    'light_fog',
    'dense_fog',
    'rain'
]
ROOTDIR = 'data/SemanticSTF'
TRAINTEXTDIR = ROOTDIR + '/train/train.txt'


with open(TRAINTEXTDIR, 'r') as f:
    train_list = f.readlines()
    train_list = [x.strip() for x in train_list]
    sample_idx, sample_cor = [], []
    for i in range(len(train_list)):
        sample_idx.append(train_list[i].split(',')[0])
        sample_cor.append(train_list[i].split(',')[1])

def get_semanticstf_info(cor, sample_idx, sample_cor):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'SemanticSTF'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'sequences/00/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'sequences/000/labels/000000.labbel',
                    'sample_id': '00'
                },
                ...
            }
        }

    sample_idx, sample_cor: list of sample index and correspoding corruption
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='SemanticSTF')
    data_list = []
    for j in range(0, len(sample_idx)):
        if cor != 'train' and sample_cor[j] != cor:
            continue
        data_list.append({
            'lidar_points': {
                'lidar_path':
                osp.join('train/velodyne', sample_idx[j] + '.bin'),
                'num_pts_feats': 4
            },
            'pts_semantic_mask_path':
            osp.join('train/labels', sample_idx[j] + '.label'),
            'sample_id': str(0) + str(j)
        })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(pkl_prefix, save_path):
    """Create info file of SemanticSTF dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info of SemanticSTF.')
    save_path = Path(save_path)
    
    for cor in cor_list:
        semanticstf_infos_train = get_semanticstf_info(cor, sample_idx, sample_cor)
        if cor == 'train':
            filename = save_path / f'{pkl_prefix}_infos_train.pkl'
        else:
            filename = save_path / f'{pkl_prefix}_infos_train_{cor}.pkl'
        # import pdb; pdb.set_trace()
        print(f'SemanticSTF info train file is saved to {filename}')
        mmengine.dump(semanticstf_infos_train, filename)