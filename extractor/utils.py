from collections import Counter


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
