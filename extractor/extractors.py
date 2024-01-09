import logging
import os

from keybert import KeyBERT
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.pipelines import pipeline


class BERTExtractor:
    def __init__(self) -> None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logging.info(f"Current device: {device}")

        BERT_JP_MODEL = os.environ.get("BERT_JP_MODEL")
        print(f"Using {BERT_JP_MODEL}")
        logging.info(f"Using {BERT_JP_MODEL}")
        hf_model = pipeline("feature-extraction", model=BERT_JP_MODEL, device=device)
        self.kw_model = KeyBERT(model=hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_JP_MODEL)

    def tokenize(self, lst_text: str = None) -> str:
        """
        Tokenizes the text
        """
        lst_text_tokenized = []
        for t in lst_text:
            inputs = self.tokenizer(t, return_tensors="pt")
            lst_text_tokenized.append(self.tokenizer.decode(inputs["input_ids"][0]))
        return lst_text_tokenized

    def extract(self, lst_text_tokenized: list = None) -> list:
        """
        Extracts the keyphrases from the tokenized text
        """
        from stop_words import STOP_WORDS_JP

        kw = self.kw_model.extract_keywords(
            lst_text_tokenized, keyphrase_ngram_range=(7, 7), stop_words=STOP_WORDS_JP
        )
        return kw
