from utils.preprocessing import prepare_text_with_regex


def test_prepare_text_with_regex():
    test_string = "Abc Def [hij] 20s."
    prepared_string = prepare_text_with_regex(test_string)

    assert prepared_string == "abc def"

