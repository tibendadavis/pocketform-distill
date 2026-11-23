from pocketform_distill.token_mapping import BYTE_MAP, bytes_to_unicode, to_student_id


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {"Ġhello": 42}
        self.encodings = {"x": [7], "xy": [7, 8]}

    def convert_tokens_to_ids(self, piece):
        return self.vocab.get(piece, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return self.encodings.get(text, [])


def test_bytes_to_unicode_is_a_full_unique_table():
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256
    for b in range(ord("!"), ord("~") + 1):
        assert table[b] == chr(b)


def test_bytes_match():
    assert BYTE_MAP[ord(" ")] == "Ġ"
    assert to_student_id({"token": " hello", "bytes": list(b" hello")}, FakeTokenizer()) == 42


def test_text_fallback():
    assert to_student_id({"token": "x", "bytes": list(b"zz")}, FakeTokenizer()) == 7
    assert to_student_id({"token": "x"}, FakeTokenizer()) == 7


def test_none_when_text_encodes_to_several_ids():
    assert to_student_id({"token": "xy"}, FakeTokenizer()) is None
