# Code for the Master's Thesis "Language Conditioned Meta-Learning for Multi-Manipulation Tasks" by Richard Bornemann
We explore the feasibility of distilling the high-level reasoning abilities exhibited by Large Language Models (LLMs), including in-context-learning, into a multi-task, language-conditioned visuomotor policy for robotics using standard behavior cloning. To this end, we introduce a novel language-based meta-learning environment **Meta-CALVIN** for robotic multi-manipulation tasks, adapted from the **CALVIN** environment, featuring two meta-learning settings: ***Explore*** and ***Combi-Explore***, designed to challenge agents with varying difficulty levels. We provide multiple datasets of expert demonstrations, collected via a LLM-based oracle agent, supporting both end-to-end training of language-conditioned visuomotor policies for control and LLM instruction tuning for robotics. We train multiple policies on the expert datasets and present promising evidence demonstrating that the high-level reasoning abilities of LLMs can be distilled into smaller, end-to-end visuomotor control policies using simple behavior cloning, enabling meta-learning agents to understand and reason about ambiguous multi-turn language feedback.


# Installation Guide and Examples for Running Data Creation, Training and Evaluation
Datasets and trained models will be released here!

## Installation
To create the venv meta-calvin run:
```
conda env create -f environment.yml
```

### Step 1)
Clone the [CALVIN](https://github.com/mees/calvin) repo. 

### Step 2)
Download the released [OpenFlamingo](https://github.com/mlfoundations/open_flamingo) models:

|# params|Language model|Vision encoder|Xattn interval*|COCO 4-shot CIDEr|VQAv2 4-shot Accuracy|Avg Len|Weights|
|------------|--------------|--------------|----------|-----------|-------|-----|----|
|3B| anas-awadalla/mpt-1b-redpajama-200b-dolly | openai CLIP ViT-L/14 | 1 | 82.7 | 45.7 | 4.09 | [Link](https://huggingface.co/openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct)|

### Step 3)
Download the RoboFlamingo [checkpoint](https://huggingface.co/roboflamingo/RoboFlamingo/blob/main/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth)

All other required Hugginface models should install automatically if they are not already in the local Huggingface Hub.
In case of manual installation they can be found here:\
Gemma-2-27b [huggingface](https://huggingface.co/google/gemma-2-27b-it)\
CLIP Vit-L/14 [huggingface](https://huggingface.co/openai/clip-vit-large-patch14)\
MPT [huggingface](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b-dolly)


To instantiate the base RoboFlamingo model:
```
from models.factory import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="PATH/TO/LLM/DIR",
    tokenizer_path="PATH/TO/LLM/DIR",
    cross_attn_every_n_layers=1,
    decoder_type='lstm',
)
```

## Data Creation
Example script for dataset generation of 7 objs Explore dataset.
```
python -u create_data.py
    --checkpoint_path 
    --use_reasoning 
    --provide_feedback 
    --rollout_steps 1200 
    --num_trials 5  
    --data_episodes 1120
    --save_dir # Path to store dataset
    --report_interval 2 
    --one_trial_limit 15
    --two_trial_limit 30 
    --zero_trial_limit 5 
    --num_objs 7 
    --seed 77 
    --reset_robot_pos
```
The actual batch scripts used to create the datasets can be found in the data gen folder

## Training
To train the model on e.g. the 7 objs Explore dataset on a single GPU run:

```
torchrun --nnodes=1 --nproc_per_node=1 --master_port=6043 train_meta_calvin.py  
    --calvin_dataset #Path to dataset
    --run_name #Run name 
    --rgb_pad 10 
    --gripper_pad 4 
    --use_gripper 
    --traj_cons 
    --fusion_mode post
    --window_size 12 
    --cross_attn_every_n_layers 1 
    --gradient_accumulation_steps 1 
    --batch_size_calvin 12 
    --precision fp32
    --num_epochs 3
    --decoder_type lstm 
    --checkpoint_path_foundation #Path to OpenFlamingo or pretrained RoboFlamingo model
    --warmup_steps 1000 
    --learning_rate 1e-5 
    --save_every_iter 1000
    --dist
```

## Evaluation
Example script of evaluating a trained model on the 7 objs Explore dataset:
```
python -u evaluate.py \
    --model_checkpoint #Path to checkpoint \
    --decoder_type lstm \
    --window_size 12 \
    --num_objs 7 \
    --num_trials 5 \
    --num_resets 15 \
    --run_name #Run name \
    --path_to_results #Path to results csv \
    --eval \
    --seed 1 \
```
Batch scripts for all our evaluation results can be found under eval

### Example Videos for trained models

### Explore Setting
<div style="display: flex; gap: 100px;">
  <img src="https://github.com/RBorn02/Embodied-Meta-Lang/raw/main/Videos/7_objs_explore_1.gif" width="300">
  <img src="https://github.com/RBorn02/Embodied-Meta-Lang/raw/main/Videos/7_objs_explore_2.gif" width="300">
</div>

### Combi Explore Setting

<div style="display: flex; gap: 100px;">
  <img src="https://github.com/RBorn02/Embodied-Meta-Lang/raw/main/Videos/combi_explore_1.gif" width="300">
  <img src="https://github.com/RBorn02/Embodied-Meta-Lang/raw/main/Videos/combi_explore_2.gif" width="300">
</div>




#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)

#### OpenFlamingo
Original: [https://github.com/mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo)
License: [MIT](https://github.com/mlfoundations/open_flamingo/blob/main/LICENSE)

#### RoboFlamingo
Original: [https://github.com/RoboFlamingo/RoboFlamingo](https://github.com/RoboFlamingo/RoboFlamingo?tab=readme-ov-file)
License:  [MIT](https://github.com/RoboFlamingo/RoboFlamingo/blob/main/LICENSE)
