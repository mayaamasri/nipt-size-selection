"""Reader labels for the identifiers used in the CSVs and the code, shared by the figures, the LaTeX tables and the
captions. The CSVs and the code keep their identifiers (W_fixedFF, shift_p10, sens_at_fpr, 1000000, ...); everything
a reader sees uses these labels. Plain text with unicode (− for minus, – for ranges); math, where needed, as $...$
(rendered by matplotlib mathtext, by LaTeX after latex_tables.tex_math, and by Markdown viewers).

check_text() is the shared self-test: no old identifier (OLD_LABELS) and no underscore outside $...$ math.

    method("W_fixedFF") -> "W (FF 10%)"       method("B_130") -> "B 40–130"
    scenario("shift_m10") -> "−10 bp"         scenario("snoise_k15") -> "$k = 1.5$"
    noise("low") -> "low noise"               basis("sens_at_fpr") -> "matched FPR"
"""

import re

METHODS = {"W_fixedFF": "W (FF 10%)", "W_G1": "W (G1 weights)"}
PLAIN_METHODS = ("A", "B", "B-opt", "W", "C1", "C2")
NOISE = {"none": "no noise", "low": "low noise", "realistic": "realistic noise"}
BASIS = {"sens": "own threshold", "sens_at_fpr": "matched FPR", "sens_lo": "conservative"}
VERSION = {"test": "test", "test+ref": "test + reference"}
FEATURES = {"zA": "$z_A$", "z_short": r"$z_\text{short}$", "z_weighted": "$z_W$"}

# Identifiers that must not reach a reader, matched outside $...$ math ('\\?' also catches the LaTeX-escaped form).
# The cell patterns (G1, noise levels as a whole table cell) let prose like 'the G1 profile' or 'low depth' pass;
# the last two catch a bare noise level as a figure title ('Low (σ = ...') and the old cutoff legend ('156 bp (B)';
# a bare '156 bp' is allowed, since it is also an axis position, e.g. Fig. 2's upper cutoff c).
OLD_LABELS = [r"W\\?_fixedFF", r"W\\?_G1", r"\bB\\?_(?:\d|c\b)", r"shift\\?_[mp]\d", r"\bblend50\b", r"snoise\\?_k",
              r"\bzA\b", r"z\\?_short", r"z\\?_weighted", r"\bsens(?:\\?_at\\?_fpr|\\?_lo)?\b", r"test\s*\+\s*ref\b",
              r"nobody\\?_positive", r"\b\d{6,}\b", r"(?:^|&)\s*G1\s*(?:&|\\\\)",
              r"(?:^|&)\s*(?:none|low|realistic)\s*(?:&|\\\\)", r"^(?:Low|Realistic|None)\s*\(",
              r"^\d{3} bp \(B\)$"]


def method(m):
    """Reader label of a method; A, B, B-opt, W, C1, C2 stay as they are, B_c is 'B 40–c'."""
    m = str(m)
    if m in METHODS:
        return METHODS[m]
    c = re.fullmatch(r"B_(\d+)", m)
    if c:
        return f"B 40–{c.group(1)}"
    if m not in PLAIN_METHODS:
        raise ValueError(f"no reader label for method {m!r}")
    return m


def cutoff(c, b_cutoff=156):
    """Reader label of the window 40–c bp: 'B 40–130'; the main window (40–156 bp) is just 'B', as in Table S10c."""
    return "B" if int(c) == b_cutoff else f"B 40–{int(c)}"


def scenario(name):
    """'G1'/'main' -> 'G1 (measured)'; 'shift_m10' -> '−10 bp'; 'blend50' -> '50% maternal blend';
    'snoise_k15' -> '$k = 1.5$'."""
    name = str(name)
    if name in ("G1", "main"):
        return "G1 (measured)"
    m = re.fullmatch(r"shift_([mp])(\d+)", name)
    if m:
        return f"{'−' if m.group(1) == 'm' else '+'}{m.group(2)} bp"
    m = re.fullmatch(r"blend(\d+)", name)
    if m:
        return f"{m.group(1)}% maternal blend"
    m = re.fullmatch(r"snoise_k(\d)(\d*)", name)
    if m:
        return f"$k = {m.group(1)}{'.' + m.group(2) if m.group(2) else ''}$"
    raise ValueError(f"no reader label for scenario {name!r}")


def noise(level):
    return NOISE[str(level)]


def noise_title(level, sigma):
    """Panel title: 'No noise (σ = 0%)', 'Low noise (σ = 0.15%)'."""
    text = noise(level)
    return f"{text[0].upper()}{text[1:]} (σ = {100 * sigma:g}%)"


def basis(name, title=False):
    text = BASIS[str(name)]
    return text[0].upper() + text[1:] if title else text


def outside_math(text):
    """The text with every $...$ segment removed (what is set as text, not math)."""
    return "".join(seg for i, seg in enumerate(str(text).split("$")) if i % 2 == 0)


def old_labels(text):
    """Old identifiers found outside math in text (each line on its own, so the cell patterns can anchor at ^)."""
    plain = outside_math(text)
    return sorted({m.group(0).strip() for line in plain.splitlines() for pat in OLD_LABELS
                   for m in re.finditer(pat, line)})


def check_text(text, where, allow_escaped=False):
    """Self-test for reader-facing text: no old label, no underscore outside math (allow_escaped: a LaTeX '\\_' is
    allowed, e.g. in a file name)."""
    old = old_labels(text)
    if old:
        raise ValueError(f"{where}: old labels {old}")
    plain = outside_math(text)
    raw = re.search(r"(?<!\\)_" if allow_escaped else "_", plain)
    if raw:
        raise ValueError(f"{where}: underscore outside math mode: {plain[max(0, raw.start() - 30):raw.end() + 30]!r}")


def self_test():
    assert method("W_fixedFF") == "W (FF 10%)" and method("B_130") == "B 40–130" and method("B-opt") == "B-opt"
    assert scenario("shift_m10") == "−10 bp" and scenario("shift_p5") == "+5 bp" and scenario("blend50") == \
        "50% maternal blend" and scenario("snoise_k15") == "$k = 1.5$" and scenario("snoise_k2") == "$k = 2$"
    assert cutoff(130) == "B 40–130" and cutoff(156) == "B"
    assert noise_title("low", 0.0015) == "Low noise (σ = 0.15%)" and noise_title("none", 0) == "No noise (σ = 0%)"
    for bad in ("W_fixedFF", "Low (σ = 0.15%)", "156 bp (B)", "shift_p10 (shift +10 bp)", "x_y", "1000000", "test+ref"):
        try:
            check_text(bad, "self-test")
        except ValueError:
            continue
        raise AssertionError(f"check_text accepted {bad!r}")
    for good in ("W (G1 weights)", "Low noise (σ = 0.15%)", "B 40–130", "$z_A$", "lines: model · points: simulation"):
        check_text(good, "self-test")


if __name__ == "__main__":
    self_test()
    print("labels self-test: ok")
