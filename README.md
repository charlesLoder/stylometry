# Fine tuning, prompting, and stylomerty

This repo contains a comparison of different strategies for affecting an LLM's output style, with a special attention given towards application in library science.

## set up

Clone this repo and using [`uv`](https://docs.astral.sh/uv/) run

```bash
uv sync
```

Add the following environment variables:

```bash
export PYTHONPATH=".venv/bin/python"
export HF_TOKEN="" # optional
```
## what is this

The repo contains notebooks for creating and comparing different strategies for affecting an LLM's text output.

Special attention is given to the field of library science, using IIIF metadata from [Northwestern University's Digital Collections](https://dc.library.northwestern.edu/)

### fine tuning

The first two notebooks are related to fine tuning.

The first notebook creates dataset from a digital collection, and the second notebook walks through the process of fine tuning an open source model.

### style profile

The third notebook generates a prompt that is informed by the field of stylometry.

### comparison

The last notebook is a comparison of:

- the base model
- using the stylistically informed prompt
- the fine tuned model

## Huggingface

It is not necessary to have a Huggingface account, but there are optional steps for pushing the dataset and fine tuned model to the Huggingface Hub.

## Testing

To test prompts and models, use the `inference.py` script:

```bash
uv run inference.py \
--model=HuggingFaceTB/SmolVLM-Instruct \
--image=https://iiif.dc.library.northwestern.edu/iiif/3/1ece48b0-8a49-491d-9f3f-90dc8bcca1ac/full/\!300,300/0/default.jpg
```