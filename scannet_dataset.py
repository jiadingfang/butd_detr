# Scannet dataset that includes both 2d and 3d data
import os
from pathlib import Path

from torch.utils.data import Dataset

from src.joint_det_dataset import Joint3DDataset
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNet, ScanNetDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset

# As we do not require any training, we can use dataparser_outputs directly

# scannet_processed_path = '/home/fjd/data/scannet/scannet_full_processed/scans/scene0000_00'
# scannet_dataparser_config = ScanNetDataParserConfig(data=Path(scannet_processed_path))
# print('scannet_dataparser_config: ', scannet_dataparser_config)
# scannet_dataparser = scannet_dataparser_config.setup()
# print('scannet_dataparser: ', scannet_dataparser)
# scannet_dataparser_outputs = scannet_dataparser.get_dataparser_outputs(split="train")
# import pdb; pdb.set_trace()
# scannet_dataparser_outputs.cameras.camera_to_worlds.shape: [5021, 3, 4]
# print('scannet_dataparser_outputs: ', scannet_dataparser_outputs)
# scannet_dataset = InputDataset(scannet_dataparser_outputs)
# print('scannet_dataset: ', scannet_dataset)
# scannet_datasample = scannet_dataset[0]
# print('scannet_datasample keys: ', scannet_datasample.keys())

class ScannetDataset(Dataset):
    def __init__(self, referit3d_datapath, scannet_processed_path, split='train'):
        self.split = split
        self.referit3d_datapath = referit3d_datapath
        self.scannet_processed_path = scannet_processed_path
        self.scannet_3d_dataset = Joint3DDataset(split=split, data_path=self.referit3d_datapath)

    def __len__(self):
        return len(self.scannet_3d_dataset)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        # 3d data
        scene_3d_data = self.scannet_3d_dataset[idx]
        scan_ids = scene_3d_data['scan_ids']

        # 2d data
        scene_2d_datapath = os.path.join(self.scannet_processed_path, scan_ids)
        scannet_dataparser_config = ScanNetDataParserConfig(data=Path(scene_2d_datapath))
        scannet_dataparser = scannet_dataparser_config.setup()
        scannet_dataparser_outputs = scannet_dataparser.get_dataparser_outputs(split=self.split)
        scene_2d_data = {
            'image_filenames': scannet_dataparser_outputs.image_filenames,
            'cameras': scannet_dataparser_outputs.cameras,
            'scene_box': scannet_dataparser_outputs.scene_box,
            'dataparser_scale': scannet_dataparser_outputs.dataparser_scale,
            'dataparser_transform': scannet_dataparser_outputs.dataparser_transform,
            'metadata': scannet_dataparser_outputs.metadata,
        }
        scene_3d_data.update(scene_2d_data)
        return scene_3d_data


if __name__ == '__main__':
    scannet_dataset = ScannetDataset(referit3d_datapath='/home/fjd/data/referit3d/', scannet_processed_path='/home/fjd/data/scannet/scannet_full_processed/scans/', split='val')
    test_data = scannet_dataset[0]
    # import pdb; pdb.set_trace()
    print('test_data keys: ', test_data.keys())
    