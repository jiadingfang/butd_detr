# outline:
# get 3d bounding boxes for all objects in the scene
# caption_lst = []
# For i in N objects:
#     get 3d bounding box for object i
#     caption_lst_i = []
#     For j in K camera views:
#         get 2d bounding box i from 3d bounding box i by projection from camera j
#         make the 2d bounding box i axis aligned and resize it to 224x224
#         use the image caption module to get a caption for the resized image
#         caption_lst_i.append(caption_j)
#     use GPT to get a summization of the captions in caption_lst_i as caption_i
#     add geometric information of 3d bounding box i to caption_i
#     caption_lst.append(caption_i)
#
# given a text query and caption_lst, use GPT to find the best object that matches the query

# util functions
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# function to get 8 corners from 3d bounding boxes
def get_eight_corners_from_3d_bboxes(bboxes):
    # Inputs
    #     bboxes: [N, 6] # cx, cy, cz, w, h, d
    # Outputs
    #     box_corners: [N, 8, 3]
    assert bboxes.shape[1] == 6 # [N, 6]
    box_corner_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ])

    bboxes_first_corner = bboxes[:, :3] - bboxes[:, 3:6] / 2 # [N, 3]
    bboxes_first_corner = np.expand_dims(bboxes_first_corner, axis=1) # [N, 1, 3]
    bboxes_first_corner = np.tile(bboxes_first_corner, [1, 8, 1]) # [N, 8, 3]
    box_corners = bboxes_first_corner + np.expand_dims(bboxes[:, 3:6], axis=1) * np.expand_dims(box_corner_vertices, axis=0) # [N, 8, 3]
    return box_corners

# function to get 2d bounding box from set of 2d points
def get_2d_bbox_from_2d_points(points_2d):
    # Inputs
    #     points_2d: [N, 2]
    # Outputs
    #     bbox_2d: [4] # x1, y1, x2, y2
    assert points_2d.shape[1] == 2
    x1 = np.min(points_2d[:, 0])
    y1 = np.min(points_2d[:, 1])
    x2 = np.max(points_2d[:, 0])
    y2 = np.max(points_2d[:, 1])
    bbox_2d = np.array([x1, y1, x2, y2])
    return bbox_2d

# get scannet 2d and 3d data
from scannet_dataset import ScannetDataset
scannet_dataset = ScannetDataset(referit3d_datapath='/home/fjd/data/referit3d/', scannet_processed_path='/home/fjd/data/scannet/scannet_full_processed/scans/', split='val')

# loading blip2 image captioning model
from lavis.models import load_model_and_preprocess
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
blip2_model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
print('blip2 model loaded!')

# Main loop
scene_idx = 0
scene_data = scannet_dataset[scene_idx]
print('scene_data keys: ', scene_data.keys()) # dict_keys(['box_label_mask', 'center_label', 'sem_cls_label', 'size_gts', 'scan_ids', 'point_clouds', 'utterances', 'positive_map', 'relation', 'target_name', 'target_id', 'point_instance_label', 'all_bboxes', 'all_bbox_label_mask', 'all_class_ids', 'distractor_ids', 'anchor_ids', 'all_detected_boxes', 'all_detected_bbox_label_mask', 'all_detected_class_ids', 'all_detected_logits', 'is_view_dep', 'is_hard', 'is_unique', 'target_cid', 'image_filenames', 'cameras', 'scene_box', 'dataparser_scale', 'dataparser_transform', 'metadata'])

import pdb; pdb.set_trace()

# get 3d bounding boxes for all objects in the scene
all_3d_bboxes = scene_data['all_bboxes'] # [132,6] # cx, cy, cz, w, h, d
all_3d_bboxes_eight_corners = get_eight_corners_from_3d_bboxes(all_3d_bboxes) # [132, 8, 3]

for obj_idx in range(len(all_3d_bboxes)):
    obj_3d_bbox = all_3d_bboxes[obj_idx] # [6]
    obj_3d_bbox_eight_corners = all_3d_bboxes_eight_corners[obj_idx] # [8, 3]
    print('obj_3d_bbox shape: ', obj_3d_bbox.shape)
    print('obj_3d_bbox_eight_corners shape: ', obj_3d_bbox_eight_corners.shape)

    caption_lst = []
    # For j in K camera views:
    #     get 2d bounding box i from 3d bounding box i by projection from camera j
    #     make the 2d bounding box i axis aligned and resize it to 224x224
    for cam_idx in range(len(scene_data['cameras'])):
        # get image from the camera view
        image_path = scene_data['image_filenames'][cam_idx]
        raw_image = Image.open(image_path).convert('RGB')
        print('image_path: ', image_path)
        print('raw_image shape: ', raw_image.size)
        
        # scene_data['cameras'].camera_to_worlds [B, 3, 4]
        cam2world_3x4 = scene_data['cameras'].camera_to_worlds[cam_idx] # [3, 4]
        # add a row [0, 0, 0, 1] to cam2world
        cam2world_4x4 = np.concatenate([cam2world_3x4, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
        world2cam_4x4 = np.linalg.inv(cam2world_4x4) # [4, 4]
        world2cam_3x4 = world2cam_4x4[:3, :] # [3, 4]
        K = scene_data['cameras'].get_intrinsics_matrices()[cam_idx] # [3, 3]
        # project 8 corners to 2d
        # repeat world2cam_3x4 8 times
        world2cam_3x4 = np.expand_dims(world2cam_3x4, axis=0) # [1, 3, 4]
        world2cam_3x4 = np.tile(world2cam_3x4, [8, 1, 1]) # [8, 3, 4]
        # project obj_3d_bbox_eight_corners to 2d with world2cam_3x4 and K
        # add a dimension at the end for obj_3d_bbox_eight_corners
        obj_3d_bbox_eight_corners = np.expand_dims(obj_3d_bbox_eight_corners, axis=-1) # [8, 3, 1]
        R, t = world2cam_3x4[:, :, :3], world2cam_3x4[:, :, 3:]
        obj_proj_eight_corners = np.matmul(K, np.matmul(R, obj_3d_bbox_eight_corners) + t) # [8, 3, 1]
        obj_proj_eight_corners = obj_proj_eight_corners[:, :2, 0] # [8, 2]
        obj_2d_bbox = get_2d_bbox_from_2d_points(obj_proj_eight_corners) # [4]

        # draw obj_proj_eight_corners as points raw image with green color
        draw = ImageDraw.Draw(raw_image)
        for point in obj_proj_eight_corners:
            draw.point(point, fill='green')
        # draw obj_2d_bbox as rectangle on raw image with red color
        draw.rectangle(obj_2d_bbox, outline='red')
        # save raw image with 2d bbox and 8 corners
        raw_image.save('raw_image_with_2d_bbox_and_8_corners.png')
        
        # crop image from raw image by obj_2d_bbox
        image = raw_image.crop(obj_2d_bbox)
        print('cropped image shape: ', image.size)
