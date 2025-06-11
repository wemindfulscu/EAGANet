# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
#-l 1 -i 1 -a 1 -s 1
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
#-l 1 -i 1 -a 1 -s 1
# ------------------------------------------------------------------------
#-l 1 -i 1 -a 1 -s 1
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

# Base Configuration
lr = 1e-4
lr_backbone = 1e-5
lr_linear_proj = 1e-5
weight_decay = 1e-4
batch_size = 1
num_workers = 1
seed = 42
max_epoch = 50
lr_drop = 40
eval_every_epoch = 1
save_every_epoch = 1
p_power = 0.5
g_power = 0.5
num_select = 100
nms_threshold = 0.5


# Model Configuration
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
random_refpoints_xy = False
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = 'relu'
dec_layer_number = None
num_body_points = 1
num_head_points = 3


# Text Encoder Configuration
text_encoder_type = "bert-base-uncased"
text_encoder_lr = 1e-5
text_encoder_frozen_start = False
text_encoder_lora = False


# Box Refinement Configuration
bbox_embed_diff_each_layer = False


# Loss Configuration
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
focal_gamma = 2.0


# Other Settings
fix_size = False
ema_decay = 0.999
use_ema = True
output_dir = "logs/DINO/R50-MS4-20e"
find_unused_params = True
sub_sentence_present = True
max_text_len = 256
num_classes = 80 # for COCO. But it is not used in Grounding DINO.

# Dataset Configuration (not used for inference, but part of the original config)
dataset_file = "coco"
coco_path = "data/coco"
remove_difficult = False

# Post-processing (not used for inference, but part of the original config)
post_process = "groundingdino_post_process"