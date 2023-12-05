from datetime import datetime, timedelta
import logging
import os
import time

from keybert import KeyBERT
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.pipelines import pipeline

from utils import preprocess, postprocess, post_ids_formatter

logging.basicConfig(level=logging.INFO)


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


def main():
    """
    This function is used to extract keyphrases from the posts
    Returns:
        List of post ids with keyphrases from the selected date
    """
    selected_date = (datetime.now() - timedelta(hours=24)).strftime("%Y%m%d")
    print(f"Selected date: {selected_date}")
    selected_date = "20231125"
    fn_train = f"data/input/{selected_date}_posts.parquet"
    df_train = pd.read_parquet(fn_train)
    train_text = df_train.loc[df_train["post_type"] == "text", "text"].tolist()
    df_train.info()

    df_train_processed = preprocess(df_train)
    df_text_sample = df_train_processed.sample(300, random_state=112).loc[
        :, ["post_id", "text"]
    ]
    lst_text_sample = df_text_sample["text"].tolist()

    BERT = BERTExtractor()
    lst_text_tokenized = BERT.tokenize(lst_text_sample)
    lst_text_tokenized = postprocess(lst_text_tokenized)

    t0 = time.time()
    kw = BERT.extract(lst_text_tokenized)
    logging.info(f"Elapsed time: {time.time() - t0:.2f} seconds")

    df_keyphrases = pd.DataFrame(
        [max(s, key=lambda x: x[1]) if s else [] for s in kw],
        columns=["keyphrases", "score"],
    )
    df_results = df_text_sample.join(df_keyphrases.set_index(df_text_sample.index))
    df_results["url_post"] = df_results["post_id"].apply(
        lambda x: f"https://yay.space/post/{x}"
    )
    logging.info(f"Have {df_results.dropna().shape[0]} posts with keyphrases")
    df_results.dropna().to_csv(
        f"data/output/csv/{selected_date}_posts_keyphrases.csv", index=False
    )
    df_results.dropna().to_parquet(
        f"data/output/parquet/{selected_date}_final_posts.parquet"
    )

    col_selected = ["post_id", "ng_score", "decline_in_public"]
    df_output = df_results.dropna()
    # uncomment this to select other dates
    selected_date = "20231105"
    df_output = pd.read_parquet(
        f"data/output/parquet/{selected_date}_final_posts.parquet"
    )
    df_pred = pd.read_parquet(f"data/inference/inference_{selected_date}_posts.parquet")
    df = pd.merge(df_output, df_pred.loc[:, col_selected], on="post_id", how="inner")

    post_ids_formatter(df_input=df)


if __name__ == "__main__":
    main()
