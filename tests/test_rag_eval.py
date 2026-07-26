from eval.rag_eval import _DIGIT_RE, evenly_spaced_sample


def test_evenly_spaced_sample_returns_all_items_when_n_exceeds_length():
    items = [1, 2, 3]
    assert evenly_spaced_sample(items, 10) == items


def test_evenly_spaced_sample_returns_requested_count_and_spans_the_list():
    items = list(range(100))
    sample = evenly_spaced_sample(items, 10)
    assert len(sample) == 10
    assert sample[0] == 0
    assert sample[-1] < 100
    assert sample == sorted(sample)  # preserves order


def test_digit_re_extracts_first_1_to_5_digit_from_judge_response():
    assert _DIGIT_RE.search("5").group() == "5"
    assert _DIGIT_RE.search("I'd rate this a 4 out of 5.").group() == "4"
    assert _DIGIT_RE.search("Score: 3\nReasoning: ...").group() == "3"


def test_digit_re_returns_none_when_no_valid_digit_present():
    assert _DIGIT_RE.search("no numeric rating here") is None
    assert _DIGIT_RE.search("0 or 6 are out of range") is None
