# Questioning the Questions: Sustaining Self-Evolution in Reasoning Models

> **R-Quest** sustains reasoning-model self-evolution by improving the questions used for training through validity and task-novelty feedback.

[Project Page](https://jinyuanli0012.github.io/R-Quest/) · [Validity-RL Data](https://huggingface.co/datasets/jinyuan222/rzero-validity-rl-terra-v1-clean-v1) · [Paper](https://arxiv.org/abs/2610.04299)

[Jinyuan Li](https://sites.google.com/view/jinyuanli)<sup>1</sup>, [Chengsong Huang](https://chengsong-huang.github.io/)<sup>1</sup>, [Langlin Huang](https://shrango.github.io/)<sup>1</sup>, [Donghong Cai](https://ilikevegetable.github.io/)<sup>1</sup>, [Shiping Gao](https://gaoshiping.github.io/)<sup>2</sup>, [Yuyi Yang](https://yyuyi.github.io/)<sup>1</sup>, [Jiaxin Huang](https://teapot123.github.io/)<sup>1,*</sup>

<sup>1</sup> Washington University in St. Louis · <sup>2</sup> University of Michigan, Ann Arbor · <sup>*</sup> Corresponding author

## 🔥 Updates

- Our [paper](https://arxiv.org/abs/2610.04299) is now available on arXiv.
- The [project page](https://jinyuanli0012.github.io/R-Quest/) presents our method, benchmark results, and ten-round self-evolution analysis.

## 🧩 Overview

Self-evolving reasoning models learn from questions they generate themselves. However, repeated self-training can eventually degrade performance. Our study identifies two problems with the generated training questions:

**Question Validity:** invalid questions provide unreliable training signals, and their growing prevalence can undermine self-evolution.

**Task Novelty:** questions can repeat the same mathematical task despite different wording, escaping lexical repetition penalties.

R-Quest addresses both through question-quality feedback. We first initialize the Solver with **Validity-RL**, teaching it to answer valid questions and reject invalid ones. During self-evolution, the evolving Solver provides validity feedback, while a frozen Base model compares each candidate with **K=8** other questions sampled from the current batch to detect repeated task types. These signals guide Questioner rewards and the construction of Solver training data.

R-Quest reaches its best average mathematical performance at **round ten (51.92%)**, exceeding R-Zero by **17.32 percentage points at the same round**. The [project page](https://jinyuanli0012.github.io/R-Quest/#results) includes the full results and transfer to code and general reasoning.

## ⚡️ Quickstart Guide

### 1. Configure Environment

The default training setup uses **4 GPUs with 80 GB memory each**.

```bash
git clone https://github.com/JinYuanLi0012/R-Quest.git
cd R-Quest

conda create -n rquest python=3.10 -y
conda activate rquest

pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation

export STORAGE_PATH=/absolute/path/to/rquest-storage
source env.sh
```

### 2. Initialize the Solver with Validity-RL

You can use a pretrained Validity-RL checkpoint or train the initial Solver yourself.

**Option A: Use the pretrained checkpoint**

Download our [Validity-RL checkpoint](https://huggingface.co/jinyuan222/R-Quest-Qwen3-4B-Validity-RL):

```bash
huggingface-cli download jinyuan222/R-Quest-Qwen3-4B-Validity-RL \
  --local-dir "$STORAGE_PATH/models/R-Quest-Qwen3-4B-Validity-RL"

export RQUEST_INITIAL_SOLVER="$STORAGE_PATH/models/R-Quest-Qwen3-4B-Validity-RL"
```

<details>
<summary><b>Option B: Train the initial Solver yourself</b></summary>

We use [this Validity-RL dataset](https://huggingface.co/datasets/jinyuan222/rzero-validity-rl-terra-v1-clean-v1). Each question is paired with a verified mathematical answer or `INVALID`.

Run the following command to initialize Qwen3-4B-Base with Validity-RL:

```bash
bash train_validity.sh
```

The next stage automatically uses the resulting checkpoint.

</details>

### 3. Run R-Quest Training

```bash
bash train_rquest.sh
```

This runs five rounds of alternating Questioner/Solver training. For ten rounds:

```bash
RQUEST_ROUNDS=10 RQUEST_RUN_NAME=rquest_qwen3_4b_k8_r10 bash train_rquest.sh
```

Training settings are in [`configs/rquest.sh`](configs/rquest.sh). Checkpoints are saved under `$STORAGE_PATH/models/`.

### 4. Resume Training

Use the same settings as the original run and add `--resume`:

```bash
bash train_rquest.sh --resume
```

See [training details](docs/TRAINING.md) for more options.

### 5. Evaluation

We use [R-Zero's evaluation code](https://github.com/Chengsong-Huang/R-Zero/tree/main/evaluation) to evaluate our models.

Set your API key for answer checking, then evaluate a trained Solver checkpoint:

```bash
export OPENAI_API_KEY=your_api_key
bash evaluation/evaluate.bash /path/to/solver/actor/huggingface
```

Results are saved under `$STORAGE_PATH/evaluation/`.

## 🎓 Acknowledgements

Our implementation builds on [R-Zero](https://github.com/Chengsong-Huang/R-Zero), [EasyR1](https://github.com/hiyouga/EasyR1), and [VERL](https://github.com/volcengine/verl). We thank their authors for releasing the training infrastructure that supports this work.

## 💬 Citation

If our work is useful for you, please consider citing our paper:

```bibtex
@misc{li2026rquest,
  title={Questioning the Questions: Sustaining Self-Evolution in Reasoning Models},
  author={Jinyuan Li and Chengsong Huang and Langlin Huang and Donghong Cai and Shiping Gao and Yuyi Yang and Jiaxin Huang},
  year={2026},
  eprint={2610.04299},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.04299}
}
```
