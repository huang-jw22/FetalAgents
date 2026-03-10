# Auxiliary Environments

`main.py` runs the orchestrator in one environment, but several bundled tools
run in separate Python environments. 


## 1. `fetal_base`

Routed here by `main.py`:

- AoP-SAM via `tools/aop_sam_step2_predict_agent.py`
- UPerNet AoP via `tools/upernet_aop_predict_agent.py`
- nnUNet HC via `tools/nnunet_hc_seg_predict_agent.py`
- nnUNet stomach segmentation via `tools/nnunet_stomach_seg_predict_agent.py`
- GA-RadImageNet via `external_tools/ga_radimagenet/`
- GA-ConvNeXt via `external_tools/ga_convnext/`

Additional traced bundled directories:

- `external_tools/AoP_SAM/`
- `external_tools/UperNet/`

Reference package versions:

```text
torch==2.5.1
torchvision==0.20.1
Pillow==10.0.1
numpy==1.26.2
opencv-python==4.8.1.78
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
SimpleITK==2.3.1
timm==0.9.16
monai==1.4.0
nnunetv2==2.6.3
einops==0.7.0
tqdm==4.66.1
scikit-image==0.22.0
h5py==3.10.0
batchgenerators==0.25.1
fvcore==0.1.5.post20221221
hausdorff==0.2.6
pycocotools==2.0.7
```

## 2. `fetalclip`

Routed here by `main.py`:

- GA-FetalCLIP via `external_tools/ga_fetalclip/`
- Plane FetalCLIP via `external_tools/plane_fetalclip/`
- Brain subplane FetalCLIP via `external_tools/brain_subplane_fetalclip/`
- Brain subplane ResNet via `tools/resnet_cls_predict_agent.py`
- Brain subplane ViT via `tools/vit_cls_predict_agent.py`
- Stomach segmentation (FetalCLIP) via `tools/stomach_seg_fetalclip_predict_agent.py`

Additional traced bundled directories:

- `external_tools/FetalCLIP_seg_stomach/`
- `external_tools/fetalclip_pred_stomach/`

Reference package versions:

```text
torch==2.0.1+cu117
torchvision==0.15.2+cu117
Pillow==11.3.0
numpy==1.26.4
opencv-python==4.10.0.84
pandas==2.3.3
timm==0.9.7
open-clip-torch==2.26.1
albumentations==2.0.8
pytorch-lightning==2.3.1
torchmetrics==1.4.0.post0
segmentation-models-pytorch==0.3.4
matplotlib==3.9.4
```

## 3. `fetalclip2`

Routed here by `main.py`:

- Video key-frame detection via `tools/video_keyframe_cls6_predict_agent.py`
- Stomach segmentation (FetalCLIP + SAMUS) via `tools/stomach_seg_fetalclip_samus_predict_agent.py`
- Abdomen segmentation (FetalCLIP) via `tools/abdomen_seg_fetalclip_predict_agent.py`
- Abdomen segmentation (FetalCLIP + SAMUS) via `tools/abdomen_seg_fetalclip_samus_predict_agent.py`

Additional traced bundled directories:

- `external_tools/keyframe_cls6/`
- `external_tools/fetalclip_pred_ac/`
- `external_tools/fetalclip_pred_stomach/`
- `external_tools/FetalCLIP_seg_ac/`
- `external_tools/FetalCLIP_seg_stomach/`

Reference package versions:

```text
torch==2.8.0
torchvision==0.23.0
Pillow==11.3.0
numpy==2.0.2
opencv-python==4.12.0.88
pandas==2.3.3
timm==1.0.20
lightning==2.5.5
pytorch-lightning==2.5.5
jsonargparse==4.41.0
peft==0.17.1
open-clip-torch==3.2.0
matplotlib==3.9.4
seaborn==0.13.2
torchmetrics==1.8.2
segmentation-models-pytorch==0.5.0
rich==14.2.0
einops==0.8.1
tqdm==4.67.1
```

## 4. `experiment_aaai`

Routed here by `main.py`:

- CSM HC measurement via `external_tools/CSM_hc/`
- FU-LoRA plane classification via `external_tools/plane_fulora/`

Reference package versions:

```text
torch==2.1.0+cu121
torchvision==0.16.0+cu121
Pillow==9.4.0
numpy==1.24.4
opencv-python==3.4.16.59
pandas==2.0.0
scipy==1.9.1
scikit-learn==1.2.2
matplotlib==3.7.1
SimpleITK==2.2.1
timm==0.9.12
```

## 5. `USFM`

Routed here by `main.py`:

- USFM AoP via `external_tools/USFM_aop/`
- USFM HC via `external_tools/USFM_hc/`

Reference package versions:

```text
torch==2.4.1+cu118
torchvision==0.19.1+cu118
Pillow==11.0.0
numpy==2.0.2
opencv-python==4.12.0.88
pandas==2.3.2
scipy==1.13.1
scikit-learn==1.6.1
matplotlib==3.9.4
SimpleITK==2.5.2
timm==1.0.19
lightning==2.5.5
hydra-core==1.3.2
omegaconf==2.3.0
einops==0.8.1
termcolor==3.1.0
tqdm==4.67.1
mmengine==0.10.7
mmsegmentation==1.2.2
torchmetrics==1.8.2
hausdorff==0.2.6
albumentations==2.0.8
typing-extensions==4.12.2
```


## 6. Configure `main.py`

After creating or reusing your environments, export the Python paths:

```bash
export FETALAGENT_FETAL_BASE_PYTHON=/path/to/envs/fetal_base/bin/python
export FETALAGENT_FETALCLIP_PYTHON=/path/to/envs/fetalclip/bin/python
export FETALAGENT_FETALCLIP2_PYTHON=/path/to/envs/fetalclip2/bin/python
export FETALAGENT_EXPERIMENT_AAAI_PYTHON=/path/to/envs/experiment_aaai/bin/python
export FETALAGENT_USFM_PYTHON=/path/to/envs/USFM/bin/python
export FETALAGENT_NNUNET_PREDICT=/path/to/envs/fetal_base/bin/nnUNetv2_predict
```
