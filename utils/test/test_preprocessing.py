from utils.preprocessing import prepare_text_with_regex


def test_prepare_text_with_regex():
    test_string = "abc [asdnpa]"
    prepared_string = prepare_text_with_regex(test_string)

    assert prepared_string == "abc "

