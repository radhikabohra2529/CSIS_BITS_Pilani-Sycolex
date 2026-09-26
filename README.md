CSIS_BITS_PILANI at SYCOLEX 2026
Code for the submission of team CSIS_BITS_Pilani to Task 2 of the track
"LLM as a Judge? From Statute Prediction to Sycophancy Detection in Law" at
FIRE 2026 — response-centric detection of sycophancy in legal case reasoning.
The system is a RoBERTa-base encoder fine-tuned over the pair of model
responses for one case under opposite assertions. Each response is prefixed with
a normalised marker naming only the side its prompt asserted
(`asserted petitioner:` / `asserted respondent:`), never the prompt text, so the
classifier reads what the model answered rather than how it was asked. This
is what allows it to transfer to the test set's entirely unseen prompt variants
(P3G–P3R). Responses longer than the 512-token limit are truncated head-and-tail,
and response order is randomised during training.
Official result: 2nd of 8 teams, macro-F1 76.81% at 83.00% accuracy.
> \*\*Note on the benchmark.\*\* SycoLex is the shared task organisers' dataset and
> is \*\*not\*\* redistributed here. Obtain it from the FIRE 2026 track organisers.
> This repository contains only our system code and our own annotations.
---
Setup
```bash
git clone https://github.com/radhikabohra2529/SycoLex.git
cd SycoLex
pip install -r requirements.txt
```
All experiments in the paper ran on a single NVIDIA GeForce RTX 4060 Laptop GPU
(8 GB) in bf16. Nothing here needs more than that.
---
Pipeline
1. Flatten the benchmark
Unpack the SycoLex release to `./sycolex`, then:
```bash
python flatten\_sycolex.py --root ./sycolex --out ./data
```
This converts the nested JSON tree into four files in `./data`:
File	Contents
`pairs.jsonl`	one row per (jurisdiction, case, model, variant) pair — the classification instances
`stance.jsonl`	one row per response (2× pairs), input to the two-stage stage-1 model
`eval\_gold.jsonl`	the subset of `pairs.jsonl` carrying a human label — the evaluation set
`flatten\_report.txt`	coverage and label diagnostics
None of these are committed; regenerate them from your own copy of the benchmark.
2. Train the detector
Validation model, held out on the Consumer Court jurisdiction to match the test
distribution:
```bash
python pair\_model.py train \\
    --data ./data --out ./pair\_model\_roberta \\
    --model roberta-base --precision bf16 \\
    --val-jurisdiction india\_consumer\_post2025
```
```bash
python pair\_model.py evaluate --data ./data --model-dir ./pair\_model\_roberta
```
`evaluate` reports F1 against the human labels on `eval\_gold.jsonl`, broken
down by jurisdiction and by prompt variant.
Training labels use the LLM-judge label with the human label overriding it where
the two disagree; validation is scored against human labels alone.
3. Pick the decision threshold
```bash
python score\_labels.py --model-dir ./pair\_model\_roberta \\
    --test task\_2\_sycophancy\_detection.jsonl --labels labels\_all.csv
```
Sweeps thresholds against the 40 hand-annotated test items (`labels\_all.csv`,
included) and reports the predicted positive rate on the full test set at each.
This is how the submitted threshold of 0.64 was chosen.
4. Final model and submission
Once all design choices were frozen, the model was retrained on every labelled
row with no holdout, carrying the threshold over rather than re-tuning it:
```bash
python pair\_model.py train \\
    --data ./data --out ./pair\_model\_final \\
    --model roberta-base --precision bf16 \\
    --final --final-threshold 0.64
```
The two submitted runs differ only in threshold:
```bash
# Run 1 (t = 0.50) — 81.33 acc, 75.94 macro-F1
python run\_test.py --mode model --model-dir ./pair\_model\_final \\
    --test task\_2\_sycophancy\_detection.jsonl \\
    --out submission\_run1.jsonl --threshold 0.50

# Run 2 (t = 0.64) — 83.00 acc, 76.81 macro-F1  \[best run, 2nd place]
python run\_test.py --mode model --model-dir ./pair\_model\_final \\
    --test task\_2\_sycophancy\_detection.jsonl \\
    --out submission\_run2.jsonl --threshold 0.64
```
`run\_test.py` handles the test file's different schema: `side\_one`/`side\_two`
order is randomised and no asserted side is given, so the asserted winner is
recovered from each prompt by a minimal-pair diff (the two prompts differ only
in the asserted party). P3R inverts, since it asserts a side lost; P3H has
identical prompts, asserts no side, and therefore cannot be sycophantic by the
task definition.
A model-free `--mode heuristic` is also available as a safety net.
---
Reproducing the paper's analyses
Paper section	Command
§3.3 label reliability (68.3% agreement, κ ≈ 0.35)	`python kappa\_calc.py`
§6.2 backbone comparison	`pair\_model.py train` with `--model` set to `roberta-base`, `xlm-roberta-base`, `law-ai/InLegalBERT`, `google/muril-base-cased`
§6.3 two-stage decomposition	`python train\_stage1.py --data ./data --out ./stage1\_model --val-jurisdiction india\_consumer\_post2025` then `python two\_stage\_eval.py --data ./data --stage1 ./stage1\_model`
§6.3 ensemble	`python ensemble\_eval.py --data ./data --models ./pair\_model\_roberta ./pair\_model\_xlmr ./pair\_model\_inlegal ./pair\_model\_muril`
§6.3 focal loss	add `--focal-gamma 2.0` to `pair\_model.py train`
§6.4 leave-one-variant-out	`python run\_lovo.py --data ./data --model roberta-base`
§5.3 annotation study	`python compare\_submissions.py --labels labels\_all.csv --subs submission\_run1.jsonl submission\_run2.jsonl`
`kappa\_calc.py` reads `data/stance.jsonl` and compares `human\_agreement` against
the LLM judge's `agreement` on the rows where both exist.
---
Our annotations
`labels\_all.csv` holds the 40 test instances we hand-annotated against the task
definition (the test labels are not public). These back the per-variant
sycophancy rates in Section 5.3 and Figure 3, and are the basis for the
threshold choice in step 3. Columns: `id`, `label` (1 = sycophantic).
The sample is stratified and deliberately weighted toward the variants expected
to be most and least sycophantic, so it is not a uniform sample of the test
set and the per-variant rates carry wide uncertainty at this size.
---
Files
```
flatten\_sycolex.py      nested SycoLex JSON -> flat JSONL
pair\_model.py           the detector: train / evaluate / predict
run\_test.py             submission generation from the released test file
score\_labels.py         threshold selection against the hand labels
train\_stage1.py         two-stage: per-response "did it cave?" classifier
two\_stage\_eval.py       two-stage: compose per-response scores into pair labels
ensemble\_eval.py        soft-voting ensemble over trained backbones
run\_lovo.py             leave-one-variant-out driver
kappa\_calc.py           human vs LLM-judge agreement and Cohen's kappa
compare\_submissions.py  score submissions against the hand labels
labels\_all.csv          our 40 hand annotations
```
---
Citation
```bibtex
@inproceedings{bohra2026sycolex,
  title     = {CSIS\\\_BITS\\\_PILANI at SYCOLEX 2026: Response-Centric Detection
               of Sycophancy in Legal Case Reasoning},
  author    = {Bohra, Radhika and Sharma, Yashvardhan},
  booktitle = {Working Notes of FIRE 2026 -- Forum for Information Retrieval
               Evaluation},
  year      = {2026}
}
```
License
MIT. See `LICENSE`.
