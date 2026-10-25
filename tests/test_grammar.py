import numpy as np
import pytest

from drishti.formats import FormatGrammar

ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
S = 10


def onehot(text: str, p: float = 0.97, confuse: dict[int, tuple[str, float]] | None = None) -> np.ndarray:
    """Slot-OCR style probability matrix for `text`, optionally with a confusion at given slots."""
    probs = np.full((S, len(ALPHA)), (1 - p) / (len(ALPHA) - 1))
    padded = text + "_" * (S - len(text))
    for i, ch in enumerate(padded):
        probs[i, :] = (1 - p) / (len(ALPHA) - 1)
        probs[i, ALPHA.index(ch)] = p
    for i, (ch, q) in (confuse or {}).items():
        probs[i] *= (1 - q) / probs[i].sum()
        probs[i, ALPHA.index(ch)] += q
        probs[i] /= probs[i].sum()
    return np.log(probs)


@pytest.fixture(scope="module")
def india():
    return FormatGrammar.from_pack("india", ALPHA, "_")


@pytest.mark.parametrize("plate,fmt", [
    ("KA05MX3021", "IN-STD"), ("MH12DE1433", "IN-STD"), ("HR85A0398", "IN-STD"), ("DL1CR9541", "IN-STD"),
    ("DL3CAB1234", "IN-STD"), ("21BH1234AB", "IN-BH"), ("22BH0001C", "IN-BH"), ("KAVAAB0123", "IN-VA"),
    ("15B123456X", "IN-ARMY"), ("77CD12", "IN-DIPLO"),
])
def test_valid_plates_decode_unchanged(india, plate, fmt):
    r = india.decode(onehot(plate), mode="strict")
    assert r.text == plate
    assert r.fmt == fmt
    assert r.format_penalty < 1e-6
    assert india.is_valid(plate) == fmt


def test_letter_o_in_digit_slot_is_corrected(india):
    # model prefers 'O' over '0' in the RTO slot; the grammar knows it must be a digit
    lp = onehot("GJ02CG3862", confuse={2: ("O", 0.6)})
    r = india.decode(lp, mode="strict")
    assert r.raw_text == "GJO2CG3862"
    assert r.text == "GJ02CG3862"
    # the look-alike mass counts as support, so confidence stays high
    assert r.prob > 0.7


def test_series_avoids_I_and_O(india):
    lp = onehot("KA05MI3021", confuse={7: ("I", 0.55)})
    lp[7, ALPHA.index("L")] = np.log(0.4)
    r = india.decode(lp, mode="strict")
    assert r.text[7] != "I"


def test_four_digit_number_recovers_truncated_read(india):
    # last digit weakly visible: free decoding drops it, the format restores it
    lp = onehot("HR85A0398", confuse={8: ("_", 0.7)})
    r = india.decode(lp, mode="strict")
    assert r.raw_text == "HR85A039"
    assert r.text == "HR85A0398"


def test_state_code_must_be_real(india):
    lp = onehot("XZ01AB1234")
    r = india.decode(lp, mode="strict")
    assert r.text[:2] != "XZ"


def test_auto_mode_falls_back_for_foreign_plate(india):
    r = india.decode(onehot("5AU5341"), mode="auto", fallback_nats=7.0)
    assert r.fmt == "free" and r.text == "5AU5341"


def test_empty_read_has_no_confidence(india):
    r = india.decode(onehot(""), mode="auto")
    assert r.text == "" and r.prob < 1e-6


def test_alternatives_ranked(india):
    lp = onehot("KA05MX3021", confuse={9: ("7", 0.35)})
    r = india.decode(lp, top_k=5)
    texts = [t for t, _ in r.alternatives]
    assert texts[0] == r.text
    assert "KA05MX3027" in texts
    confs = [c for _, c in r.alternatives]
    assert confs == sorted(confs, reverse=True)


def test_brazil_pack():
    g = FormatGrammar.from_pack("brazil", ALPHA, "_")
    assert g.is_valid("ABC1D23") == "BR-MERCOSUR"
    assert g.is_valid("ABC1234") == "BR-LEGACY"
    lp = onehot("ABC1D23", confuse={3: ("I", 0.6)})  # '1' read as 'I' where only a digit fits
    assert g.decode(lp).text == "ABC1D23"
    # 'D' vs '0' at slot 4 is a genuine ambiguity: both layouts are valid, the evidence decides
    lp = onehot("ABC1D23", confuse={4: ("0", 0.6)})
    assert g.decode(lp).text == "ABC1023"


def test_custom_pack_dict():
    pack = {"name": "t", "classes": {"D": {"chars": "0-9"}, "L": {"chars": "A-Z"}},
            "patterns": [{"name": "X", "pattern": "L{2} D{2,3}"}]}
    g = FormatGrammar.from_pack(pack, ALPHA, "_")
    assert {t.length for t in g.templates} == {4, 5}
    assert g.decode(onehot("AB12")).text == "AB12"
