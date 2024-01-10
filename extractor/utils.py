import csv
from collections import Counter
import logging
import re

import pandas as pd


def get_higest_percentage(text: str = None) -> (str, int, float):
    """
    Returns the highest percentage of a character in a text
    """
    char_counter = Counter()
    lst_text = [s for s in text if s not in ["\n"]]
    if lst_text:
        char_counter.update(lst_text)
        len_text = sum(char_counter.values())
        cnt_most_common = char_counter.most_common(1)[0][1]
        str_most_common = char_counter.most_common(1)[0][0]
    else:
        str_most_common = None
        cnt_most_common = 0
        len_text = 1
    return (str_most_common, cnt_most_common, cnt_most_common / len_text)


def preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the text
    """
    LEN_POST_UPPER = 104
    LEN_POST_LOWER = 8
    PCT_CHAR_UPPER = 0.36
    df = df_input.loc[
        (df_input["post_type"] == "text") & (~df_input["text"].str.contains("#"))
    ].copy()
    df["len_text"] = df["text"].apply(len)
    df = df.join(
        pd.DataFrame(
            df.loc[:, "text"].apply(get_higest_percentage).tolist(),
            columns=["top_char", "cnt_char", "pct_char"],
        )
    )
    df = df.loc[
        (df["len_text"] < LEN_POST_UPPER)
        & (df["len_text"] > LEN_POST_LOWER)
        & (df["pct_char"] < PCT_CHAR_UPPER),
        :,
    ].copy()
    print(f"Output DataFrame shape: {df.shape}")
    return df.reset_index(drop=True)


def postprocess(tokenized_texts: list = None) -> list:
    """
    Postprocesses the tokenized text
    - remove digits
    - remove english characters
    """
    lst_text_cleaned = []
    for tkn in tokenized_texts:
        tmp_tkn = tkn.split(" ")
        prefix = tmp_tkn[0]
        suffix = tmp_tkn[-1]
        tkn = [re.sub(r"[a-zA-Z0-9]", "", s) for s in tmp_tkn[1:-1]]
        tkn = list(dict.fromkeys(tkn))
        lst_text_cleaned.append(" ".join([prefix] + tkn + [suffix]))
    return lst_text_cleaned


def get_keyphrases(df_input: pd.DataFrame = None, kw: list = None) -> pd.DataFrame:
    """
    Returns a DataFrame of posts with keyphrases and scores.
    Args:
        df_input: DataFrame of posts
        kw: list of keyphrases and scores from KeyBERT
    Output:
        DataFrame of posts with keyphrases and scores
    """
    df_keyphrases = pd.DataFrame(
        [max(s, key=lambda x: x[1]) if s else [] for s in kw],
        columns=["keyphrases", "score"],
    )
    df = df_input.join(df_keyphrases.set_index(df_input.index))
    df["url_post"] = df["post_id"].apply(lambda x: f"https://yay.space/post/{x}")
    df = df.dropna()
    logging.info(f"Have {df.shape[0]} posts with keyphrases")
    return df


def post_ids_formatter(
    df_input: pd.DataFrame = None,
    mode: str = None,
) -> pd.DataFrame:
    """
    Format DataFrame of keywords posts and save to CSV file
    Args:
        df_input: DataFrame of posts with keywords
        mode: normal or textai
    Output:
        Post IDs in CSV format
    """
    LEN_LINE_OUTPUT = 10
    fn_csv = "data/output/results/post_ids.csv"
    if not mode:
        raise ValueError("Mode is not specified")
    elif mode == "textai":
        raise NotImplementedError
        THRESHOLD_NG_SCORE = 0.8
        THRESHOLD_DECLINE_IN_PUBLIC = 0.9
        df_final_post_ids = df_input.loc[
            (df_input["ng_score"] <= THRESHOLD_NG_SCORE)
            & (df_input["decline_in_public"] <= THRESHOLD_DECLINE_IN_PUBLIC),
            :,
        ]
        lst_final_post_ids = df_final_post_ids["post_id"].tolist()
    else:
        lst_final_post_ids = df_input["post_id"].tolist()
    print(f"Number of final posts: {len(lst_final_post_ids)}")
    with open(fn_csv, "w") as f:
        writer = csv.writer(f)
        for idx in range(len(lst_final_post_ids) // LEN_LINE_OUTPUT):
            idx_start = idx * LEN_LINE_OUTPUT
            idx_end = (idx + 1) * LEN_LINE_OUTPUT
            writer.writerow(lst_final_post_ids[idx_start:idx_end])

        writer.writerow(lst_final_post_ids[idx_end:])
    logging.info(f"Post IDs are saved in {fn_csv}")
