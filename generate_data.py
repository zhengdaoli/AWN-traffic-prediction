import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def feature_array(frame, add_time_of_day, add_day_of_week):
    values = np.expand_dims(frame.values.astype(np.float32), axis=-1)
    features = [values]
    if add_time_of_day:
        fraction = (frame.index.values - frame.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        features.append(np.tile(fraction[:, None, None], (1, frame.shape[1], 1)).astype(np.float32))
    if add_day_of_week:
        day = frame.index.dayofweek.to_numpy(dtype=np.float32) / 6.0
        features.append(np.tile(day[:, None, None], (1, frame.shape[1], 1)))
    return np.concatenate(features, axis=-1)


def make_samples(data, history_length, prediction_length, context_mode, points_per_day):
    inputs = []
    targets = []
    first_origin = history_length
    if context_mode == "periodic":
        first_origin = 7 * points_per_day
    for origin in range(first_origin, len(data) - prediction_length + 1):
        recent = data[origin - history_length:origin]
        if context_mode == "periodic":
            weekly_start = origin - 7 * points_per_day
            daily_start = origin - points_per_day
            weekly = data[weekly_start:weekly_start + history_length]
            daily = data[daily_start:daily_start + history_length]
            sample = np.concatenate([weekly, daily, recent], axis=0)
        else:
            sample = recent
        inputs.append(sample)
        targets.append(data[origin:origin + prediction_length, :, 0])
    return np.stack(inputs), np.stack(targets)


def split_and_save(inputs, targets, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    total = len(inputs)
    train_end = round(total * 0.7)
    validation_end = train_end + total - round(total * 0.2) - train_end
    ranges = {"train": (0, train_end), "val": (train_end, validation_end), "test": (validation_end, total)}
    for name, (start, end) in ranges.items():
        np.savez_compressed(output_directory / f"{name}.npz", x=inputs[start:end], y=targets[start:end])
        print(name, inputs[start:end].shape, targets[start:end].shape)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_length", type=int, default=12)
    parser.add_argument("--prediction_length", type=int, default=12)
    parser.add_argument("--context_mode", choices=("contiguous", "periodic"), default="contiguous")
    parser.add_argument("--points_per_day", type=int, default=288)
    parser.add_argument("--add_time_of_day", action="store_true")
    parser.add_argument("--add_day_of_week", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    frame = pd.read_hdf(arguments.traffic_file)
    data = feature_array(frame, arguments.add_time_of_day, arguments.add_day_of_week)
    x, y = make_samples(data, arguments.history_length, arguments.prediction_length, arguments.context_mode, arguments.points_per_day)
    split_and_save(x, y, Path(arguments.output_dir))
