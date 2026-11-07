"""Pins for the Kronos checkout this integration was written against.

Weights are not stored in HedgeFundBot. Revisions are Hugging Face commit
SHAs so a later upload cannot silently change the model that a run records.
"""

from __future__ import annotations

KRONOS_REPO = "https://github.com/shiyu-coder/Kronos"
# master on 2026-04-13, the tree whose ``KronosPredictor.predict`` was read
# for this integration. ``sample_count`` averages decoded paths; it does not
# return an interval. See ``predict.py``.
REVIEWED_UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

# Kronos-mini is the 4.1M model. Its tokenizer is Kronos-Tokenizer-2k, not
# Kronos-Tokenizer-base (that one belongs to small/base). Context length 2048.
MODEL_ID = "NeoQuasar/Kronos-mini"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
MAX_CONTEXT = 2048

LICENSE_NAME = "MIT"
LICENSE_NOTICE = "Kronos is MIT licensed, Copyright (c) 2025 ShiYu. It is not relicensed under HedgeFundBot's Apache-2.0 terms."
PAPER_URL = "https://arxiv.org/abs/2508.02739"

# The paper (arXiv:2508.02739v1, 2025-08-02), the upstream README, and the
# NeoQuasar/Kronos-mini and Kronos-Tokenizer-2k model cards describe a
# multi-exchange pretraining corpus and do not give a sample end date.
PRETRAINING_CUTOFF = None
PRETRAINING_CONTAMINATION = "unknown"
PRETRAINING_NOTE = (
    "No pretraining sample end date is published for NeoQuasar/Kronos-mini "
    "(paper arXiv:2508.02739, Hugging Face model card, upstream README). "
    "A backtest that overlaps unknown training data is contaminated until "
    "that cutoff is known. Do not read such a score as out-of-sample skill."
)

INFLUENCE_SHADOW = "shadow"
AMOUNT_POLICY_OMIT = "omit"
AMOUNT_POLICY_APPROXIMATE = "approximate_volume_times_mean_ohlc"
CALENDAR_ID = "XNYS"
DEFAULT_TIMEZONE = "America/New_York"
# Pinned after a run against pandas 3.0.6. Holiday and early-close tests
# describe this version's XNYS schedule; bump the pin and the tests together.
EXCHANGE_CALENDARS_VERSION = "4.13.2"
DEFAULT_HORIZONS = "5,20"
DEFAULT_LOOKBACK = 256
DEFAULT_MIN_BARS = 64
DEFAULT_SEED = 123
DEFAULT_PATH_COUNT = 1
MAX_PATH_COUNT = 8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_DEVICE = "cpu"
