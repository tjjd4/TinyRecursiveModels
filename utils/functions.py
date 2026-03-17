import importlib
import inspect
import os
import torch
import torch.nn as nn


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def load_checkpoint_from_path(model: nn.Module, load_path: str):
    if not load_path:
        return

    print(f"Loading checkpoint {load_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        
        repo_id = load_path[5:]
        
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            from safetensors import safe_open
            state_dict = {}
            with safe_open(model_path, framework="pt", device=device) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        except Exception as e:
            print(f"Failed to load safetensors from HF ({e}), falling back to pytorch_model.bin")
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            state_dict = torch.load(model_path, map_location=device)
    else:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        state_dict = torch.load(load_path, map_location=device)

    # Resize and reset puzzle emb if needed
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    if hasattr(model, "model") and hasattr(model.model, "puzzle_emb"):
        expected_shape = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    if len(missing):
        print(f"Missing keys (ok if head changed): {len(missing)}")
        for k in missing[:20]:
            print("  missing:", k)
    if len(unexpected):
        print(f"Unexpected keys (ok if old head existed): {len(unexpected)}")
        for k in unexpected[:20]:
            print("  unexpected:", k)
