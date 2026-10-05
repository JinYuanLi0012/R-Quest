# Training details

## Configuration

The main settings are in `configs/rquest.sh`:

| Variable | Default | Purpose |
|---|---|---|
| STORAGE_PATH | `<checkout>/runs` | Checkpoints, datasets, and caches |
| RQUEST_INITIAL_SOLVER | Locally trained Validity-RL step 15 | Initial Solver checkpoint |
| RQUEST_RUN_NAME | `rquest_qwen3_4b_k8` | Output prefix |
| RQUEST_ROUNDS | 5 | Number of training rounds |
| RQUEST_GPU_IDS | `0,1,2,3` | Four visible GPU indices |
| RQUEST_BASE_MODEL | `Qwen/Qwen3-4B-Base` | Initial Questioner and frozen novelty judge |
| RQUEST_VALIDITY_DATASET | `jinyuan222/rzero-validity-rl-terra-v1-clean-v1` | Validity-RL training and replay data |

Each round trains the Questioner for five steps, generates 10,000 candidate questions, and trains the Solver for 15 steps. Generated questions pass Solver validity voting and an answer-consistency filter in [0.3, 0.8]. Validity-RL replay accounts for approximately 10% of the mixed Solver training data.

The backend also downloads `hiyouga/math12k` for loader rows and math validation. Questioner training replaces the problem text with the fixed question-generation prompt.

## Resume

Keep the same storage directory, run name, round count, and configuration when resuming. Preserve the generated datasets and checkpoints with the run state.

```bash
bash train_validity.sh --resume
bash train_rquest.sh --resume
```

Validity-RL resumes up to a total of 15 steps. R-Quest skips completed stages and resumes interrupted training from the latest complete checkpoint. Resuming an interrupted stage requires optimizer and dataloader states; the merged `actor/huggingface` directory is used to initialize a new stage.

Choose ten rounds before starting if you want the extended run:

```bash
RQUEST_ROUNDS=10 RQUEST_RUN_NAME=rquest_qwen3_4b_k8_r10 bash train_rquest.sh
```
