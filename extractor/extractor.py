from datetime import datetime, timedelta
import time

from keybert import KeyBERT
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.pipelines import pipeline


from stop_words import STOP_WORDS_JP
from utils import preprocess, postprocess

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Current device: {device}")


BERT_JP_MODEL = "cl-tohoku/bert-base-japanese-v3"
hf_model = pipeline("feature-extraction", model=BERT_JP_MODEL, device=device)
kw_model = KeyBERT(model=hf_model)

tokenizer = AutoTokenizer.from_pretrained(BERT_JP_MODEL)

selected_date = (datetime.now() - timedelta(hours=24)).strftime("%Y%m%d")
print(f"Selected date: {selected_date}")

# selected_date = '20231125'
fn_train = f"data/input/{selected_date}_posts.parquet"
df_train = pd.read_parquet(fn_train)
train_text = df_train.loc[df_train["post_type"] == "text", "text"].tolist()
df_train.info()

df_train_processed = preprocess(df_train)
# df_text_sample = df_train_processed.sample(500, random_state=112).loc[:, ['post_id', 'text']]
df_text_sample = df_train_processed.sample(frac=1, random_state=112).loc[
    :, ["post_id", "text"]
]
lst_text_sample = df_text_sample["text"].tolist()

lst_text_tokenized = []
for t in lst_text_sample:
    inputs = tokenizer(t, return_tensors="pt")
    lst_text_tokenized.append(tokenizer.decode(inputs["input_ids"][0]))

lst_text_tokenized = postprocess(lst_text_tokenized)

t0 = time.time()
kw = kw_model.extract_keywords(
    lst_text_tokenized, keyphrase_ngram_range=(7, 7), stop_words=STOP_WORDS_JP
)
print(f"Elapsed time: {time.time() - t0:.2f} seconds")

df_keyphrases = pd.DataFrame(
    [max(s, key=lambda x: x[1]) if s else [] for s in kw],
    columns=["keyphrases", "score"],
)
df_results = df_text_sample.join(df_keyphrases.set_index(df_text_sample.index))
df_results["url_post"] = df_results["post_id"].apply(
    lambda x: f"https://yay.space/post/{x}"
)
print(f"Have {df_results.dropna().shape[0]} posts with keyphrases")
df_results.dropna().to_csv(
    f"data/output/csv/{selected_date}_posts_keyphrases.csv", index=False
)
df_results.dropna().to_parquet(
    f"data/output/parquet/{selected_date}_final_posts.parquet"
)

col_selected = ["post_id", "ng_score", "decline_in_public"]
df_output = df_results.dropna()
# uncomment this to select other dates
# selected_date = '20231105'
df_output = pd.read_parquet(f"data/output/parquet/{selected_date}_final_posts.parquet")
df_pred = pd.read_parquet(f"data/inference/inference_{selected_date}_posts.parquet")
df = pd.merge(df_output, df_pred.loc[:, col_selected], on="post_id", how="inner")

LEN_LINE_OUTPUT = 10
THRESHOLD_NG_SCORE = 0.8
THRESHOLD_DECLINE_IN_PUBLIC = 0.9
df_final_post_ids = df.loc[
    (df["ng_score"] <= THRESHOLD_NG_SCORE)
    & (df["decline_in_public"] <= THRESHOLD_DECLINE_IN_PUBLIC),
    :,
]
lst_final_post_ids = df_final_post_ids["post_id"].tolist()
print(f"Number of final posts: {len(lst_final_post_ids)}")
print(f"From: {selected_date}")
for idx in range(len(lst_final_post_ids) // LEN_LINE_OUTPUT):
    idx_start = idx * LEN_LINE_OUTPUT
    idx_end = (idx + 1) * LEN_LINE_OUTPUT
    print(lst_final_post_ids[idx_start:idx_end])
print(lst_final_post_ids[idx_end:])
