import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

# load test image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

import time
start_time = time.time()
# load model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
# Other available models:
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )
end_time = time.time()
print('model loaded in {} seconds'.format(end_time - start_time))
print('blip2 model loaded!')

# preprocess image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

start_time = time.time()
# get caption
caption = model.generate({"image": image})
end_time = time.time()
print('caption generated in {} seconds'.format(end_time - start_time))
print(caption)

# Example output:
# model loaded in 127.94595074653625 seconds
# blip2 model loaded!
# caption generated in 5.300491094589233 seconds
# ['the merlion fountain in singapore']