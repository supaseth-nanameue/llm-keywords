from pandas import DataFrame

from extractor.utils import get_higest_percentage, preprocess, postprocess


def test_get_higest_percentage_return_tuple():
    test_text = "aaaaaaaaabbbcc"
    assert isinstance(get_higest_percentage(test_text), tuple)


def test_preprocess_return_dataframe():
    assert isinstance(preprocess(pd.DataFrame()), DataFrame)


def test_postprocess_return_list():
    test_input_text = ["something", "to", "test"]
    assert isinstance(postprocess(test_input_text), list)
