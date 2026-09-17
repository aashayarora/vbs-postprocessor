# VBS ABCDNet

## Environment Setup

Run the following commands to setup the environment for the training

```
conda create —name abcd

conda activate abcd

conda install python==3.12

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install pyaml numpy uproot matplotlib scikit-learn tqdm pytorch-lightning tensorboard pandas
```

## Config

- `sig_base_path`, `bkg_base_path`
- `sig_path`, `bkg_path`
- `training_features`
- `constraint_var`
- `extra_vars` (ensures `weight` is present)
- `derived_vars` (optional mapping of `new_var: expression`, computed before preprocessing)
- `auto_include_derived_vars` (optional bool, auto-adds derived var names to `training_features`)
- `preselection` (optional)
- `learning_rate`, `batch_size`, `n_epochs`, `output`

Optional keys:

- `derived_vars` (default `{}`), for example:

	- `delta_eta_abs: "np.abs(boosted_h_candidate_eta - boosted_v_candidate_eta)"`
	- `pt_ratio: "boosted_h_candidate_pt / (boosted_v_candidate_pt + 1e-6)"`

- `auto_include_derived_vars` (default `false`)
- `auto_include_derived_vars_transform` (default `none`), transform to apply to auto-included derived vars

- `flavor` (default `single`): `single` for classical disco or `double` for double disco
- `weight_decay` (default `1e-2`)
- `label_smoothing` (default `0.0`)
- `use_lr_scheduler` (default `true`)
- `lr_scheduler_patience` (default `10`)
- `lr_scheduler_factor` (default `0.5`)
- `lr_scheduler_min_lr` (default `1e-6`)
- `architecture` (default `[64, 32, 16]`)
- `bce_weight` (default `1.0`)
- `disco_lambda` (default `0.0`)
- `closure_lambda` (default `0.0`)
- `closure_n_events_min` (default `10`)
- `closure_symmetrize` (default `true`)
- `use_batchnorm` (default `true`)
- `dropout` (default `0.0`)

## Run

```bash
python3 main.py --config config_boosted_run3.yaml --flavor single
```

Then monitor logs with TensorBoard from the configured output directory. For example:

```bash
tensorboard --logdir <output_dir>
```
(Replace `<output_dir>` with the `output` directory specified in your config, and access `http://localhost:6006` in your browser.)

To skip training and run only the inference step (possibly with a --checkpoint definition)

```bash
python3 main.py --config single/config_boosted_run3.yaml --flavor single --infer --checkpoint <path>
```

To run inference on data after training

```bash
python3 main.py --config single/config_boosted_run3.yaml --flavor single --infer --checkpoint <path> --data
```
