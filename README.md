# Spell Check with a Sequence-to-Sequence Model

A character-level encoder-decoder (GRU with Bahdanau attention) that reads a misspelled word one character at a time and writes out the corrected spelling. It applies the classic neural machine translation architecture to spell checking: the source language is misspelled text, the target language is correct text.

## Repo structure

- `Spell_Check.ipynb` - the full walkthrough: data preparation, encoder/attention/decoder construction, training, inference, attention plots, and SavedModel export.
- `spell_correct.csv` - training pairs of misspelled and correct words.
- `big.txt`, `wikipedia.txt`, `aspell.txt`, `birkbeck.txt`, `spell-testset1.txt`, `spell-testset2.txt` - word lists and evaluation sets.
- `translator/` - an exported TensorFlow SavedModel of the trained translator (large binary, used as-is).

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Open `Spell_Check.ipynb` and run the cells in order.

Training runs for many epochs and benefits from a GPU. To skip training and just run inference, load the exported model in `translator/` with `tf.saved_model.load`.

## What you learn

- Character-level tokenization with `TextVectorization`.
- Building an encoder-decoder with Bahdanau (additive) attention in Keras.
- Masked sequence loss that ignores padding tokens.
- Teacher-forced training with a custom `train_step`.
- Autoregressive inference with sampling and attention visualization.
- Exporting a trained model as a TensorFlow SavedModel.
