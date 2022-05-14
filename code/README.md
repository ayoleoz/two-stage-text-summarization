# Environment
```
conda create -n text-sum python=3.8
conda activate text-sum
pip install -r code/requirements.txt
```

# Evaluate Baselines on test set
```
python3 code/baselines/score_baselines.py
```

# BART
## Fine-tune BART on CNN/DM
```
python3 code/bart/finetune_bart.py 
```
By default, model checkpoint will be saved to `code/bart/checkpoints/checkpoint.pt`.
Only single GPU training is supported.

## BERTSUMEXT + BART Paraphraser
```
python3 code/bart/finetune_bart.py --train-ext-output output/BertSumExt/Score-0.5/bertsumext-out-train.txt --validation-ext-output output/BertSumExt/Score-0.5/bertsumext-out-validation.txt
```
By default, model checkpoint will be saved to `code/bart/checkpoints/checkpoint.pt`.

## Evaluate on test set and dump output
```
python3 code/bart/score_bart.py --weight code/bart/checkpoints/best.pt --output-dir output/bertsumext-bart-out
```

## Evaluate on test set output (ROUGE 1/2/L F1)
```
python3 code/bart/score_output.py --output-dir output/bertsumext-bart-out
```

Use the commandline option `-h` for more information about script usage.

# GPT-2