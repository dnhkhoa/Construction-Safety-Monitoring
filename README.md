# Construction Safety Monitoring

This repository supports a university AI research project on construction safety monitoring. The current focus is a clean, reproducible object-detection baseline for identifying construction head-safety conditions from images or video frames.

## Research Motivation

Construction sites contain dynamic hazards, changing environments, and workers with varying personal protective equipment (PPE) compliance. A reliable visual baseline is needed before adding higher-level safety reasoning, behavior analysis, or reporting.

## Current Research Problem

The current baseline studies detection of two classes:

- `helmet`
- `head`

This allows initial evaluation of helmet-related PPE detection while keeping the research scope narrow and reproducible.

## Current Baseline

The baseline model is **YOLO11n** using the Ultralytics implementation.

High-level pipeline:

```text
Data -> Preprocessing -> YOLO11n -> Predictions -> Evaluation
```

## Evaluation Metrics

The baseline should be evaluated with:

- Precision
- Recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95

No experimental results are reported here until they are produced and documented in this repository.

## Repository Structure

```text
Construction-Safety-Monitoring/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── baseline_yolo11.ipynb
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   └── convert_annotations.py
│   ├── training/
│   │   └── train_yolo.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── inference/
│       └── predict.py
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── results/
│   └── .gitkeep
├── docs/
│   └── pipeline.md
└── tests/
```

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Setup

Datasets are not committed to Git. Place local data under `data/raw/` or `data/processed/` and create a YOLO dataset YAML file. See `data/README.md` for the expected structure.

Update `configs/baseline.yaml` so `dataset_yaml` points to the local dataset YAML file.

## Training

```bash
python -m src.training.train_yolo --config configs/baseline.yaml
```

or:

```bash
bash scripts/train.sh
```

## Evaluation

```bash
python -m src.evaluation.evaluate --config configs/baseline.yaml --weights results/train/weights/best.pt
```

or:

```bash
bash scripts/evaluate.sh results/train/weights/best.pt
```

## Inference

```bash
python -m src.inference.predict --config configs/baseline.yaml --weights results/train/weights/best.pt --source path/to/image_or_directory
```

## Scoped Safety Agents

The repository also carries a small, isolated reasoning package under
`src/safety_agents/`:

- `ContextAgent` validates model proposals against typed references and bounded
  actions. Its default adapter is intentionally unconfigured and returns
  `ABSTAIN` rather than inventing evidence.
- `RuleSeverityAgent` accepts only a gate result with the
  `READY_FOR_RULE` route and maps normalized helmet/zone evidence to the checked-in
  rule catalog.
- `EvidenceSufficiencyGate` and the shared typed contracts are included only as
  the minimum control boundary required by those two agents.

This port does not add PPE, Zone, Orchestrator, Reporting, or other MoA agents,
and it does not connect the agents to the baseline pipeline automatically. See
`context.md` for the project-wide ownership and fail-closed constraints.

Run the scoped contract tests with:

```bash
python -m unittest discover -s tests -v
```

## Future Extensions

Future research stages may add:

- Full PPE detection
- Behavior analysis
- Integration of the scoped agents into a controlled end-to-end flow
- Automatic safety reporting

These extensions are intentionally not implemented in the current baseline setup.
