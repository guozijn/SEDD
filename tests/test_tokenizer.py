from sedd_mini.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip_unicode():
    tokenizer = ByteTokenizer()
    text = "SEDD handles 中文 and math: p(y)/p(x)."
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == text
    assert max(ids) < tokenizer.mask_id
