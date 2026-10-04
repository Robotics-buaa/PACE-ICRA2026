"""Read-only validation of the 64-env, 30-iteration G1 wrist checkpoint."""
import argparse
from pathlib import Path


def check_finite(value, torch, path):
    if torch.is_tensor(value):
        assert bool(torch.isfinite(value).all()), "Non-finite tensor: " + path
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, torch, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_finite(item, torch, path + "[%d]" % index)


def optimizer_steps(optimizer):
    states = optimizer.get("state", {})
    assert states, "Optimizer contains no recorded parameter state"
    steps = []
    for state in states.values():
        if "step" in state:
            step = state["step"]
            steps.append(float(step.item() if hasattr(step, "item") else step))
    assert steps and max(steps) > 0, "Optimizer has no positive update step"
    return max(steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        root = Path.cwd() / "logs/g1_table_tennis_wrist"
        suffix = "_g1_wrist_pred_smoke_64env_30iter"
        runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix))
        if not runs:
            raise FileNotFoundError("No matching wrist smoke run under " + str(root))
        checkpoint = runs[-1] / "model_29.pt"
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Final checkpoint missing: " + str(checkpoint))

    import torch
    # This file was produced by the user's own local training run.
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print("Checkpoint:", checkpoint, flush=True)
    assert data["iter"] == 29, ("Unexpected iteration", data["iter"])
    print("[PASS] Final checkpoint: iter=29", flush=True)

    policy = data["model_state_dict"]
    expected = {
        "actor.0.weight": (512, 450), "actor.6.weight": (24, 128),
        "critic.0.weight": (512, 535), "critic.6.weight": (1, 128),
    }
    for key, shape in expected.items():
        assert tuple(policy[key].shape) == shape, (key, tuple(policy[key].shape), shape)
    print("[PASS] Wrist policy dimensions: actor 450->24, critic 535->1", flush=True)
    predictor = data["pred_state_dict"]
    assert predictor, "Predictor weights are missing or empty"
    for name, weights in (("policy", policy), ("predictor", predictor)):
        assert any(torch.is_tensor(value) for value in weights.values()), name + " contains no tensors"
        check_finite(weights, torch, name)
    print("[PASS] Policy and predictor weights are finite", flush=True)
    for name in ("optimizer_state_dict", "pred_optimizer_state_dict"):
        optimizer = data[name]
        check_finite(optimizer, torch, name)
        step = optimizer_steps(optimizer)
        print("[PASS] %s: recorded updates, max step=%g" % (name, step), flush=True)
    print("G1 WRIST SHORT TRAINING CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
