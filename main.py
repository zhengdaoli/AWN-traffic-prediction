import argparse
import json
from pathlib import Path

import torch

from model import AWN
from util import load_adjacency, load_config, load_datasets, metrics, set_seed


def evaluate(model, loader, scaler, device, null_value=None, clip_predictions=None):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            output = model(inputs.to(device))
            output = scaler.inverse_transform(output[..., 0])
            if clip_predictions is not None:
                output = output.clamp(clip_predictions[0], clip_predictions[1])
            predictions.append(output.cpu())
            targets.append(labels.cpu())
    return metrics(torch.cat(predictions), torch.cat(targets), null_value)


def train(config):
    set_seed(config["seed"])
    device = torch.device(config["device"] if config["device"] != "cuda" or torch.cuda.is_available() else "cpu")
    loaders, scaler = load_datasets(config["data_path"], config["batch_size"], config["seed"])
    adjacency = load_adjacency(config["adjacency_path"])
    model = AWN(
        adjacency=adjacency,
        input_channels=config["input_channels"],
        hidden_channels=config["hidden_channels"],
        output_channels=config["output_channels"],
        history_length=config["history_length"],
        prediction_length=config["prediction_length"],
        scales=config["scales"],
        rank=config["rank"],
        shift_weight=config["shift_weight"],
        graph_layers=config["graph_layers"],
        recurrent_layers=config["recurrent_layers"],
        directed=config["directed"],
        context_mode=config["context_mode"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    null_value = config.get("null_value")
    best_value = float("inf")
    best_epoch = 0
    patience_count = 0
    checkpoint = Path(config["checkpoint_path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            prediction = scaler.inverse_transform(model(inputs)[..., 0])
            loss = metrics(prediction, labels, null_value)["mae"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
        validation = evaluate(model, loaders["val"], scaler, device, null_value, config.get("clip_predictions"))
        value = validation["mae"].item()
        print(json.dumps({"epoch": epoch, **{key: item.item() for key, item in validation.items()}}))
        if value < best_value:
            best_value = value
            best_epoch = epoch
            patience_count = 0
            torch.save({"model": model.state_dict(), "config": config, "scaler": {"mean": scaler.mean, "std": scaler.std}}, checkpoint)
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                break
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_result = evaluate(model, loaders["test"], scaler, device, null_value, config.get("clip_predictions"))
    result = {"best_epoch": best_epoch, **{key: value.item() for key, value in test_result.items()}}
    print(json.dumps(result))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(load_config(arguments.config))
