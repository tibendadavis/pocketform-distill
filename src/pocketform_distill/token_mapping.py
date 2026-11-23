"""Map teacher tokens (text/bytes from Ollama) to student token ids. See README."""


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_MAP = bytes_to_unicode()


def to_student_id(entry: dict, tok) -> int | None:
    raw = entry.get("bytes")
    if raw:
        piece = "".join(BYTE_MAP[b] for b in raw)
        tid = tok.convert_tokens_to_ids(piece)
        if tid is not None and tid != tok.unk_token_id:
            return tid
    ids = tok.encode(entry["token"], add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None
