from datetime import datetime, timedelta
import logging
import pandas as pd
import time

from extractors import BERTExtractor

from utils import preprocess, postprocess, get_keyphrases
from utils import post_ids_formatter


logging.basicConfig(level=logging.INFO)


def main(mode: str = "normal", selected_date: str = None):
    """
    This function is used to extract keyphrases from the posts
    Args:
        mode: normal or textai
        date: date of posts to be extracted in YYYYMMDD format
    Returns:
        List of post ids with keyphrases from the selected date
    """
    if not selected_date:
        selected_date = (datetime.now() - timedelta(hours=24)).strftime("%Y%m%d")
    elif selected_date == "today":
        selected_date = datetime.now().strftime("%Y%m%d")
    print(f"Selected date: {selected_date}")
    fn_train = f"data/input/{selected_date}_posts.parquet"
    logging.info(f"Reading {fn_train}")
    df_train = pd.read_parquet(fn_train)
    logging.info(f"Loaded {df_train.shape[0]} posts")

    df_train_processed = preprocess(df_train)
    lst_text_sample = df_train_processed["text"].tolist()

    BERT = BERTExtractor()
    lst_text_tokenized = BERT.tokenize(lst_text_sample)
    lst_text_tokenized = postprocess(lst_text_tokenized)

    logging.info(f"Extracting keywords...")
    t0 = time.time()
    kw = BERT.extract(lst_text_tokenized)
    logging.info(f"Elapsed time: {time.time() - t0:.2f} seconds")

    df_results = get_keyphrases(df_train_processed, kw)

    df_results.to_csv(
        f"data/output/csv/{selected_date}_posts_keyphrases.csv", index=False
    )
    df_results.to_parquet(f"data/output/parquet/{selected_date}_final_posts.parquet")

    logging.info(f"Extracting post ids using {mode} mode")

    post_ids_formatter(df_input=df_results, mode=mode)


if __name__ == "__main__":
    main(mode="normal", selected_date="today")
