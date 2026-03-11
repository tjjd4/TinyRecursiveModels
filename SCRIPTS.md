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
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_44/step_65104" \
+run_name=${run_name}
```

#### Z Analysis
```bash
run_name="eval_pretrain_mlp_t_sudoku_z_analysis_78"
python z_analysis.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
load_checkpoint="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_ga_78/step_65104" \
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
load_checkpoint="checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30_1gpu_44/step_65104" \
+run_name=${run_name}
```

## TRM Pretrain Experiments

### Sudoku-Extreme (assuming 1 L40S GPU):

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

Expected: Around 87% exact-accuracy (+- 2%)

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

## HRM Pretrain Experiments

### Sudoku-Extreme

```bash
run_name="pretrain_hrm_sudoku_ga_78"
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
arch=trm_rl \
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
arch=trm_rl \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=500 eval_interval=50 \
lr=1e-6 puzzle_emb_lr=1e-6 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
load_checkpoint="checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30_1gpu_44/step_65104" \
+run_name=${run_name} ema=True
```
