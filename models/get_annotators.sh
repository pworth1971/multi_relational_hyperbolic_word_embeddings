#!/usr/bin/env bash

pip install -U gdown
gdown https://drive.google.com/uc?id=1CBYxiL3k8dHcDZYjcW_Fct_pyWyk912j
unzip distill_bert.zip -d distill_bert
rm distill_bert.zip