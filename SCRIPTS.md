# Scripts

## Evaluation

### Sudoku-Extreme:

```bash
run_name="eval_pretrain_mlp_t_sudoku"
python eval.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
+run_name=${run_name}
```

```bash
run_name="eval_pretrain_mlp_t_sudoku_incorrect_44"
python eval.py \
arch=trm \
data_paths="[data/split/sudoku-extreme-1k-aug-1000/incorrect]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
+run_name=${run_name}
```

### Maze-Hard:

```bash
run_name="eval_pretrain_att_maze30x30_1gpu"
python eval.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
load_checkpoint="checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30_1gpu_44/step_65100" \
+run_name=${run_name}
```

## Z Analysis

### Sudoku-Extreme:

```bash
run_name="z_analysis_pretrain_mlp_t_sudoku_44"
python z_analysis.py \
arch=trm_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
+run_name=${run_name}
```

#### Random Weight

```bash
run_name="z_analysis_random_weight_mlp_t_sudoku_44"
python z_analysis.py \
arch=trm_random_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name}
```

#### Reset z_L per H cycle

```bash
run_name="z_analysis_pretrain_mlp_t_sudoku_reset_zL_per_H_cycle_44"
python z_analysis.py \
arch=trm_reset_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
arch.reset_z_L_per_step=False \
arch.reset_z_L_per_H_cycle=True \
arch.reset_z_H_per_step=False \
arch.reset_z_H_per_H_cycle=False \
+run_name=${run_name}

run_name="z_analysis_pretrain_mlp_t_sudoku_reset_zH_per_H_cycle_44"
python z_analysis.py \
arch=trm_reset_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
arch.reset_z_L_per_step=False \
arch.reset_z_L_per_H_cycle=False \
arch.reset_z_H_per_step=False \
arch.reset_z_H_per_H_cycle=True \
+run_name=${run_name}
```

#### Reset z_L or z_H at specific steps

```bash
run_name="z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_6_44"
python z_analysis.py \
arch=trm_reset_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
arch.reset_z_L_per_step=False \
arch.reset_z_L_per_H_cycle=False \
arch.reset_z_H_per_step=False \
arch.reset_z_H_per_H_cycle=False \
arch.reset_z_L_at_steps="[5]" \
arch.reset_z_H_at_steps="[]" \
+run_name=${run_name}

run_name="z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_6_44"
python z_analysis.py \
arch=trm_reset_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
arch.reset_z_L_per_step=False \
arch.reset_z_L_per_H_cycle=False \
arch.reset_z_H_per_step=False \
arch.reset_z_H_per_H_cycle=False \
arch.reset_z_L_at_steps="[]" \
arch.reset_z_H_at_steps="[5]" \
+run_name=${run_name}
```

#### Data only correct or incorrect

```bash
run_name="z_analysis_pretrain_mlp_t_sudoku_correct"
python z_analysis.py \
arch=trm_trace \
data_paths="[data/split/sudoku-extreme-1k-aug-1000/correct]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
+run_name=${run_name}


run_name="z_analysis_pretrain_mlp_t_sudoku_incorrect"
python z_analysis.py \
arch=trm_trace \
data_paths="[data/split/sudoku-extreme-1k-aug-1000/incorrect]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100" \
+run_name=${run_name}
```

#### Model from HuggingFace (HRM or TRM)

Replace load_checkpoint with the path to the model checkpoint downloaded from HuggingFace

```bash
run_name="z_analysis_trm_mlp_t_sudoku_huggingface"
python z_analysis.py \
arch=trm_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku/trm-mlp-hf/checkpoint" \
+run_name=${run_name}
```

```bash
run_name="z_analysis_hrm_sudoku_huggingface"
python z_analysis.py \
arch=hrm_trace \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
load_checkpoint="checkpoints/Sudoku/hrm-hf/checkpoint" \
+run_name=${run_name}
```

## Scripts

### Split Dataset correct and incorrect

#### Sudoku-Extreme:

```bash
python split_dataset.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100"
```

##### With Rating

```bash
python split_dataset_with_rating.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100"
```

### Consistency Check

#### Sudoku-Extreme:

```bash
python scripts/consistency_check.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65100"
+num_runs=5
```

### Compute Degradation Rate

```bash
python scripts/compute_degradation_rate.py \
--baseline   checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_44/z_analysis_step_0/z_raw.npz \
--ablation_A checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_per_H_cycle_44/z_analysis_step_0/z_raw.npz \
--ablation_B checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_per_step_44/z_analysis_step_0/z_raw.npz

python scripts/compute_degradation_rate.py \
--baseline   checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_44/z_analysis_step_0/z_raw.npz \
--ablation_A checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_8_44/z_analysis_step_0/z_raw.npz \
--ablation_B checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_8_44/z_analysis_step_0/z_raw.npz
```

### Reset Analysis

```bash
python scripts/reset_analysis.py \
--baseline  checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_44/z_analysis_step_0/z_raw.npz \
--zH_k2     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_2_44/z_analysis_step_0/z_raw.npz \
--zH_k4     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_4_44/z_analysis_step_0/z_raw.npz \
--zH_k6     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_6_44/z_analysis_step_0/z_raw.npz \
--zH_k8     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_8_44/z_analysis_step_0/z_raw.npz \
--zH_k10    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_10_44/z_analysis_step_0/z_raw.npz \
--zH_k12    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_12_44/z_analysis_step_0/z_raw.npz \
--zH_k14    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zH_at_step_14_44/z_analysis_step_0/z_raw.npz \
--zL_k2     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_2_44/z_analysis_step_0/z_raw.npz \
--zL_k4     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_4_44/z_analysis_step_0/z_raw.npz \
--zL_k6     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_6_44/z_analysis_step_0/z_raw.npz \
--zL_k8     checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_8_44/z_analysis_step_0/z_raw.npz \
--zL_k10    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_10_44/z_analysis_step_0/z_raw.npz \
--zL_k12    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_12_44/z_analysis_step_0/z_raw.npz \
--zL_k14    checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/z_analysis_pretrain_mlp_t_sudoku_reset_zL_at_step_14_44/z_analysis_step_0/z_raw.npz \
--save_dir  checkpoints/Sudoku-extreme-1k-aug-1000-trace-torch/degradation_curve
```

## TRM Pretrain Experiments

### Sudoku-Extreme (assuming 1 L40S GPU):

#### MLP-T
```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

#### Attention
```bash
run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

#### Learn z0
```bash
run_name="pretrain_mlp_t_sudoku_learn_z0"
python pretrain.py \
arch=trm_learn_z0 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

## HRM Pretrain Experiments

### Sudoku-Extreme

```bash
run_name="pretrain_hrm_sudoku"
python pretrain.py \
arch=hrm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name}
```

## GRPO Experiments

### Sudoku-Extreme:

#### Outcome Supervision (GRPO-OS)
```bash
run_name="grpo_mlp_t_sudoku_DR"
python train_grpo_os.py \
arch=trm_grpo \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=500 eval_interval=50 \
lr=1e-6 puzzle_emb_lr=1e-6 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_78/step_65104" \
+run_name=${run_name} ema=True
```

### Maze-Hard:

```bash
run_name="grpo_att_maze30x30_1gpu_DR"
python train_grpo_os.py \
arch=trm_grpo \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=500 eval_interval=50 \
lr=1e-6 puzzle_emb_lr=1e-6 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
load_checkpoint="checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30_1gpu_44/step_65104" \
+run_name=${run_name} ema=True
```
