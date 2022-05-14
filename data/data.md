# CNN/Dailymail Dataset

## Data Access

This dataset is available in the [Hugging Face datasets collection](https://huggingface.co/datasets/cnn_dailymail), therefore it's preferred to resort to its API to use the data, rather than download the raw files and write our own loader.

Specifically:

1. Install the Hugging Face datasets library via pip: `pip install datasets`
2. In python:

```python
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
```

## Fields and Splits

Data has three fields:

- `id`: a string containing the heximal formated SHA1 hash of the url where the story was retrieved from
- `article`: a string containing the body of the news article, which is used as our source text
- `highlights`: a string containing the highlight of the article as written by the article author, which is used as our target summary

An example is

```json
{
    "id":"0054d6d30dbcad772e20b22771153a2a9cbeaf62",
    "article":"(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.",
    "highlights":"The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says."
}
```

The CNN/DailyMail dataset has 3 splits: train, validation, and test. Below are the number of instances in each split:

| Dataset Split | Number of Instances in Split |
| ------------- | ---------------------------- |
| Train         | 287,113                      |
| Validation    | 13,368                       |
| Test          | 11,490                       |

They can be accessed like

```python
dataset["train"]
dataset["validation"]
dataset["test"]
```

## Link to Data Dump

As the project specification requires us to provide a link to the dataset, we dumped the data and uploaded it to the Google Drive. The link is: https://drive.google.com/drive/folders/1M9M5Mr8Ka75eM95bcA3T_ZBC8TjkDm-A?usp=sharing



