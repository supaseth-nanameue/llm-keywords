from dagster import asset, op
from keybert import KeyBERT
from transformers.pipelines import pipeline

import json
import os
import requests


BERT_JP_MODEL = "cl-tohoku/bert-base-japanese-v3"

@asset
def basic_asset():
    return 5



@op
def keyword_extractor(model=BERT_JP_MODEL):
    return None
