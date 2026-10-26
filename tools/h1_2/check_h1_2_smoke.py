"""Read-only acceptance of a new 64-env, 30-iteration H1-2 checkpoint."""
import argparse
from pathlib import Path

ROOT = Path("/workspace/projects/PACE-ICRA2026/logs/h1_2_table_tennis")
SUFFIX = "_h1_2_pred_smoke_64env_30iter"


def finite(value, torch, name):
    if torch.is_tensor(value):
        assert bool(torch.isfinite(value).all()), "NONFINITE TENSOR: " + name
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, torch, name + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite(child, torch, name + "[" + str(index) + "]")


def optimizer_step(optimizer):
    states = optimizer.get("state", {})
    assert states, "OPTIMIZER STATE EMPTY"
    steps = [float(s["step"].item() if hasattr(s["step"], "item") else s["step"]) for s in states.values() if "step" in s]
    assert steps and max(steps) > 0, "NO RECORDED OPTIMIZER UPDATES"
    return max(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
    else:
        runs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.endswith(SUFFIX)) if ROOT.is_dir() else []
        assert runs, "NO MATCHING H1-2 SHORT TRAINING RUN"
        if len(runs) != 1:
            for run in runs:
                print("Candidate:", run / "model_29.pt", flush=True)
            raise RuntimeError("MULTIPLE RUNS: select the new checkpoint with --checkpoint")
        checkpoint = runs[0] / "model_29.pt"
    assert checkpoint.is_file(), "CHECKPOINT NOT FOUND: " + str(checkpoint)
    import torch
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print("Checkpoint:", checkpoint, flush=True)
    assert data["iter"] == 29, "EXPECTED FINAL ITERATION 29"
    policy = data["model_state_dict"]
    shapes = {
        "actor.0.weight": (512, 495), "actor.6.weight": (27, 128),
        "critic.0.weight": (512, 580), "critic.6.weight": (1, 128),
    }
    for key, shape in shapes.items():
        assert tuple(policy[key].shape) == shape, f"MODEL SHAPE MISMATCH: {key}"
    predictor = data["pred_state_dict"]
    assert predictor, "PREDICTOR WEIGHTS MISSING"
    matrices = [(key, value) for key, value in predictor.items() if torch.is_tensor(value) and value.ndim == 2 and key.endswith("weight")]
    assert matrices and matrices[0][1].shape[1] == 15 and matrices[-1][1].shape[0] == 3, "PREDICTOR DIMENSION MISMATCH"
    for name, weights in (("policy", policy), ("predictor", predictor)):
        assert any(torch.is_tensor(v) for v in weights.values()), "NO WEIGHT TENSORS: " + name
        finite(weights, torch, name)
    print("[PASS] iter=29; actor 495->27; critic 580->1; predictor 15->3", flush=True)
    print("[PASS] Policy and predictor weights finite", flush=True)
    for key in ("optimizer_state_dict", "pred_optimizer_state_dict"):
        finite(data[key], torch, key)
        print("[PASS]", key, "max step=", optimizer_step(data[key]), flush=True)
    print("H1-2 SHORT TRAINING CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
