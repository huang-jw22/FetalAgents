# FetalAgent: Multi-Agent Fetal Ultrasound Analysis System

A multi-agent system for automated fetal ultrasound analysis, integrating multiple specialized computer vision models for plane classification, biometry measurement, gestational age estimation, and segmentation. The system uses an LLM-based orchestrator to coordinate task allocation and report generation.

## Architecture

FetalAgent employs a **hybrid symbolic-neural** architecture:

- **Allocator Agent**: Analyzes user inquiries, performs plane classification, and dispatches tasks to appropriate expert agents.
- **Expert Agents**: Each expert wraps multiple CV tool ensembles with deterministic decision logic (weighted voting, residual-based gating, tiered fallback).
- **Report Generator**: Produces structured clinical-style reports with biometry results, growth percentile assessment, and segmentation outputs.

### Supported Tasks

| Task | Expert | Tools (Ensemble) |
|------|--------|-----------------|
| Plane Classification | plane_classification | FetalCLIP zero-shot, FU-LoRA |
| Brain Subplane Classification | brain_subplanes | FetalCLIP probe, ResNet-50, ViT |
| Head Circumference (HC) | head_circumference | CSM + nnU-Net (residual-gated) |
| Gestational Age (GA) | gestational_age | RadImageNet, FetalCLIP, ConvNeXt (weighted vote) |
| Abdomen Segmentation | abdomen_segmentation | FetalCLIP+SAMUS |
| Stomach Segmentation | stomach_segmentation | FetalCLIP, FetalCLIP+SAMUS, nnU-Net (tiered fallback) |
| Angle of Progression (AoP) | aop | AoP-SAM, USFM-AoP, UPerNet |
| Video Summary | video_summary | Key-frame detection + per-frame expert routing |

## Project Structure

```
FetalAgent/
|-- main.py                    # Core orchestration script
|-- tools/                     # Agent-side tool wrapper scripts
|-- external_tools/            # External model inference scripts
|   |-- AoP_SAM/               # AoP-SAM model code
|   |-- UperNet/               # UPerNet model code
|   |-- USFM_aop/              # USFM for AoP segmentation
|   |-- USFM_hc/               # USFM for HC segmentation
|   |-- CSM_hc/                # CSM for HC measurement
|   |-- ga_radimagenet/        # GA estimation (RadImageNet)
|   |-- ga_fetalclip/          # GA estimation (FetalCLIP)
|   |-- ga_convnext/           # GA estimation (ConvNeXt)
|   |-- plane_fetalclip/       # Plane classification (FetalCLIP)
|   |-- plane_fulora/          # Plane classification (FU-LoRA)
|   |-- brain_subplane_fetalclip/  # Brain subplane (FetalCLIP)
|   +-- keyframe_cls6/         # Video key-frame detection
|-- reference/                 # Growth reference tables
|   |-- HC_GA_reference.csv
|   +-- AC_GA_reference.csv
|-- example_images/            # Example test data
|-- requirements.txt
|-- LICENSE
+-- README.md
```

The checkpoint bundle is distributed separately as a sibling folder:

```text
../FetalAgent_ckpt/
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/huang-jw22/FetalAgents.git
cd FetalAgents
```

### 2. Set up Python environments

The system requires multiple conda environments due to different model dependencies:

```bash
# Main environment (for main.py orchestrator)
conda create -n fetalagent python=3.10
conda activate fetalagent
pip install -r requirements.txt
```

Additional conda environments are needed for specific tools:
- **hxt_base**: AoP-SAM, UPerNet, nnUNet, GA-RadImageNet, GA-ConvNeXt
- **fetalclip**: FetalCLIP-based plane/subplane/GA tools
- **fetalclip2**: SAMUS-based segmentation, video key-frame detection
- **experiment_aaai**: CSM HC measurement, FU-LoRA plane classification
- **USFM**: USFM-based AoP/HC tools

Example setup commands for these auxiliary environments are provided in
`ENVIRONMENTS.md`. If you already have compatible research environments,
you can reuse them and only set the environment variables below.

### 3. Download checkpoints

Place the released checkpoint bundle next to the code folder:

```text
/your/path/
├── FetalAgent_submission/
└── FetalAgent_ckpt/
```

The code resolves checkpoints from the sibling folder automatically.
If you store it elsewhere, set:

```bash
export FETALAGENT_CKPT_DIR=/path/to/FetalAgent_ckpt
```

### 4. Configure tool environments

You do not need to edit `main.py`.
Set the Python executables for your conda environments through environment variables:

```bash
export FETALAGENT_HXT_BASE_PYTHON=/path/to/envs/hxt_base/bin/python
export FETALAGENT_FETALCLIP_PYTHON=/path/to/envs/fetalclip/bin/python
export FETALAGENT_FETALCLIP2_PYTHON=/path/to/envs/fetalclip2/bin/python
export FETALAGENT_EXPERIMENT_AAAI_PYTHON=/path/to/envs/experiment_aaai/bin/python
export FETALAGENT_USFM_PYTHON=/path/to/envs/USFM/bin/python
export FETALAGENT_NNUNET_PREDICT=/path/to/envs/hxt_base/bin/nnUNetv2_predict
```

If you have installed everything into a single environment, you can leave these unset and the system will fall back to `python`.

### 5. Set up API key

The LLM orchestrator requires an OpenAI-compatible API:

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="gpt-5-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Single Image Analysis

```bash
# Estimate gestational age
python main.py \
    --inquiry "Estimate gestational age of this fetal ultrasound scan image." \
    --case_dir example_images/brain_thalamic

# Comprehensive caption for a fetal brain scan
python main.py \
    --inquiry "Write a comprehensive caption for this fetal ultrasound scan image." \
    --case_dir example_images/brain_thalamic

# Comprehensive caption for a fetal abdomen scan
python main.py \
    --inquiry "Write a comprehensive caption for this fetal ultrasound scan image." \
    --case_dir example_images/fetal_abdomen
```

### Video Summary

```bash
python main.py \
    --inquiry "This folder contains continuous screenshots of a fetal US video. The Gestational Age estimated from last menstrual period (LMP) is 23w 0d. Please provide a comprehensive summary." \
    --case_dir example_images/video
```

### Input Format

Each `case_dir` should contain:
- One or more ultrasound images (`.png` or `.jpg`)
- A `pixel_size.csv` file with columns: `filename,pixel size(mm)`

Example `pixel_size.csv`:
```csv
filename,pixel size(mm)
315_HC.png,0.201985378458
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This system integrates the following open-source projects:
- [FetalCLIP](https://github.com/13204942/FetalCLIP)
- [AoP-SAM](https://github.com/13204942/AoP-SAM)
- [SAMUS](https://github.com/xianlin7/SAMUS)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [AutoGen](https://github.com/microsoft/autogen)
