# AWN

PyTorch implementation of Adaptive Wavelet Graph Networks for multistep traffic forecasting.

## Requirements

- Python 3.10 or later
- PyTorch 2.2 or later
- NumPy, pandas, and PyTables

```bash
git clone https://github.com/zhengdaoli/AWN-traffic-prediction.git
cd AWN-traffic-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

METR-LA and PeMS-BAY use the splits and sensor graphs distributed with [DCRNN](https://github.com/liyaguang/DCRNN). NYCBike1 and NYCBike2 use the processed data and graph files distributed with [ST-SSL](https://github.com/Echo-Ji/ST-SSL_Dataset).

Prepared files are stored as `train.npz`, `val.npz`, and `test.npz`:

| Key | Contiguous shape | Periodic shape |
| --- | --- | --- |
| `x` | `(samples, H, nodes, features)` | `(samples, 3H, nodes, features)` |
| `y` | `(samples, P, nodes)` | `(samples, P, nodes)` |

For five-minute road-network observations:

```bash
python generate_data.py \
  --traffic_file data/metr-la.h5 \
  --output_dir data/METR-LA \
  --history_length 12 \
  --prediction_length 12 \
  --context_mode contiguous
```

`contiguous` uses the `H` observations immediately preceding the prediction origin. `periodic` concatenates weekly, daily, and recent windows in that order. The two modes are explicit and cannot be mixed in one checkpoint.

## Model

Each graph layer constructs one branch per fixed wavelet scale. For scale `s`, the implementation uses `Psi_s = exp(sL)` and `Psi_s_inv = exp(-sL)`, a trainable diagonal wavelet filter, and a scale-specific feature projection. Directed graphs use separate operators from `A` and `A.T`.

Scale weights are computed for every sample, history step, and node from the cosine similarity between a projected layer input and each projected branch representation. Softmax normalization is applied across scales.

The shifted kernel is parameterized once per model as `L1 @ L2`, where `L1` has shape `(nodes, rank)` and `L2` has shape `(rank, nodes)`. The resulting correction is shared across graph layers, wavelet scales, directions, and history steps. It is added to each wavelet branch before spatial propagation.

The encoded history initializes a multilayer GRU. The final history representation is reused as the decoder input at every forecast step while the recurrent state evolves. Predictions are not fed back into the decoder.

## Training

Dataset configurations are under `configs/`. They specify the complete scale set, rank, seed, history length, prediction length, optimizer settings, early-stopping patience, input mode, metric mask, and prediction clipping policy.

```bash
python main.py --config configs/metr_la.json
python main.py --config configs/pems_bay.json
```

The best checkpoint is selected by validation MAE and saved to the configured path. With `null_value` set to `null`, MAE and RMSE use every target value; MAPE excludes zero denominators. With a numeric `null_value`, all three metrics use the same validity mask. `clip_predictions` is `null` in the supplied configurations, so reported predictions are not clipped.

## Periodic comparison

Generate a separate dataset with `--context_mode periodic`, copy the corresponding configuration, change `data_path`, set `context_mode` to `periodic`, and use a distinct `checkpoint_path`. The graph encoder and all other model parameters remain shared across the weekly, daily, and recent contexts. Their forecasts are combined by learned node- and horizon-specific softmax weights.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
