"""
Tokenizer Efficiency & Cost Impact Analysis
=============================================
Compares how efficiently different tokenizers encode the same underlying
content across languages and domains, and translates the fragmentation
gap into a $ cost impact using published token pricing.

DESIGNED FOR: Google Colab Free (CPU-only runtime is enough).
NO PAID API / NO API KEY REQUIRED — everything here runs locally:
  - tiktoken's BPE files and the two HuggingFace tokenizers used below are
    all free, public, and ungated (no HF login/token needed).

Run top to bottom as-is. Cells are marked with `# %%` so you can also
paste this into Colab as separate cells (Colab / Jupyter auto-detects them).
"""

# %% ------------------------------------------------------------------
# 0. SETUP (run once)
# -----------------------------------------------------------------------
# In Colab, run this in its own cell first:
#
# !pip install -q tiktoken transformers sentencepiece pandas matplotlib
#
# (Everything below is pure Python / local computation after that.)

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% ------------------------------------------------------------------
# 1. LOAD TOKENIZERS
# -----------------------------------------------------------------------
# We use three tokenizers spanning different design philosophies:
#   - GPT-4 / GPT-4o family (tiktoken, cl100k_base)      -> English-centric BPE
#   - Llama 3                                             -> larger (128k) BPE
#     vocab with somewhat broader multilingual coverage
#   - A dedicated multilingual SentencePiece tokenizer     -> mT5 (101 languages)
#
# The Llama 3 tokenizer files are mirrored on HF under a non-gated repo
# (NousResearch's mirror of meta-llama/Meta-Llama-3-8B) so no HF account,
# login, or access token is required.

import tiktoken
from transformers import AutoTokenizer

print("Loading tokenizers (first run downloads a few small files)...")

gpt4_tok = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-4o

llama3_tok = AutoTokenizer.from_pretrained(
    "NousResearch/Meta-Llama-3-8B"  # ungated mirror, no token needed
)

mt5_tok = AutoTokenizer.from_pretrained(
    "google/mt5-small"  # SentencePiece, 101-language coverage, ungated
)

TOKENIZERS = {
    "GPT-4o (tiktoken/cl100k_base)": lambda t: gpt4_tok.encode(t),
    "Llama 3 (128k BPE)": lambda t: llama3_tok.encode(t, add_special_tokens=False),
    "mT5 (multilingual SentencePiece)": lambda t: mt5_tok.encode(t, add_special_tokens=False),
}

print("Tokenizers loaded:", list(TOKENIZERS.keys()))

# %% ------------------------------------------------------------------
# 2. FIXED TEXT SAMPLES (language x domain)
# -----------------------------------------------------------------------
# Same underlying meaning across English / Hindi / Swahili (health-advice
# paragraph), plus two English domain samples (legal, medical) and one
# code sample, so we can separately compare (a) language effects and
# (b) domain effects on tokenizer fragmentation.
#
# NOTE ON TRANSLATION QUALITY: the Hindi and Swahili paragraphs below are
# close paraphrases of the English one, written for this benchmark, not
# a certified/professional translation. For rigorous research, swap these
# for a verified parallel corpus such as FLORES-200
# (https://github.com/facebookresearch/flores) which gives you the same
# sentence across 200 languages.

SAMPLES = {
    "English (prose)": (
        "Regular physical activity and proper hydration play an important "
        "role in maintaining good health. Doctors recommend that adults "
        "drink at least two liters of water every day and engage in "
        "moderate exercise for thirty minutes, five times a week. This "
        "simple routine can reduce the risk of heart disease, improve "
        "mental clarity, and increase overall energy levels throughout "
        "the day."
    ),
    "Hindi (prose, same meaning)": (
        "नियमित शारीरिक गतिविधि और उचित जलयोजन अच्छे स्वास्थ्य को बनाए रखने "
        "में महत्वपूर्ण भूमिका निभाते हैं। डॉक्टर सलाह देते हैं कि वयस्कों को "
        "प्रतिदिन कम से कम दो लीटर पानी पीना चाहिए और सप्ताह में पांच बार "
        "तीस मिनट के लिए मध्यम व्यायाम करना चाहिए। यह सरल दिनचर्या हृदय रोग "
        "के खतरे को कम कर सकती है, मानसिक स्पष्टता में सुधार कर सकती है, "
        "और पूरे दिन ऊर्जा के स्तर को बढ़ा सकती है।"
    ),
    "Swahili (prose, same meaning)": (
        "Mazoezi ya mwili mara kwa mara na unywaji wa maji ya kutosha "
        "vina jukumu muhimu katika kudumisha afya njema. Madaktari "
        "wanashauri watu wazima kunywa angalau lita mbili za maji kila "
        "siku na kufanya mazoezi ya wastani kwa dakika thelathini, mara "
        "tano kwa wiki. Utaratibu huu rahisi unaweza kupunguza hatari ya "
        "magonjwa ya moyo, kuboresha uwazi wa kiakili, na kuongeza kiwango "
        "cha nishati mwilini siku nzima."
    ),
    "Python (code)": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    result = []\n"
        "    for _ in range(n):\n"
        "        result.append(a)\n"
        "        a, b = b, a + b\n"
        "    return result\n"
        "\n"
        "class Cache:\n"
        "    def __init__(self, maxsize=128):\n"
        "        self.maxsize = maxsize\n"
        "        self.data = {}\n"
        "\n"
        "    def get(self, key, default=None):\n"
        "        return self.data.get(key, default)\n"
        "\n"
        "    def set(self, key, value):\n"
        "        if len(self.data) >= self.maxsize:\n"
        "            oldest = next(iter(self.data))\n"
        "            del self.data[oldest]\n"
        "        self.data[key] = value\n"
    ),
    "Legal (English, domain)": (
        "Notwithstanding any provision to the contrary herein, the "
        "Indemnifying Party shall defend, indemnify, and hold harmless "
        "the Indemnified Party from and against any and all claims, "
        "liabilities, damages, losses, costs, and expenses, including "
        "reasonable attorneys' fees, arising out of or relating to any "
        "breach of the representations, warranties, or covenants set "
        "forth in this Agreement, except to the extent such claims arise "
        "from the gross negligence or willful misconduct of the "
        "Indemnified Party."
    ),
    "Medical (English, domain)": (
        "The patient presented with acute exacerbation of chronic "
        "obstructive pulmonary disease, manifesting as dyspnea, "
        "productive cough, and wheezing. Arterial blood gas analysis "
        "revealed hypoxemia with mild hypercapnia. Treatment was "
        "initiated with nebulized bronchodilators, systemic "
        "corticosteroids, and supplemental oxygen therapy, with close "
        "monitoring of oxygen saturation and respiratory rate."
    ),
}

# %% ------------------------------------------------------------------
# 3. TOKENIZE EVERYTHING & COMPUTE FRAGMENTATION RATIOS
# -----------------------------------------------------------------------
# word count uses whitespace splitting. This is an imperfect but standard
# proxy: Hindi and Swahili are also space-delimited so it is directionally
# fair; it is not meaningful for languages that don't use spaces
# (e.g. Chinese, Japanese, Thai) — swap in a language-appropriate word
# segmenter if you extend this to those languages.

rows = []
for category, text in SAMPLES.items():
    n_chars = len(text)
    n_words = len(text.split())
    for tok_name, tok_fn in TOKENIZERS.items():
        n_tokens = len(tok_fn(text))
        rows.append({
            "category": category,
            "tokenizer": tok_name,
            "n_chars": n_chars,
            "n_words": n_words,
            "n_tokens": n_tokens,
            "tokens_per_char": n_tokens / n_chars,
            "tokens_per_word": n_tokens / n_words,
        })

df = pd.DataFrame(rows)

# Fragmentation ratio: tokens/char for this category vs. English prose,
# for the SAME tokenizer. >1 means "more fragmented (less efficient)
# than English on this tokenizer".
baseline = (
    df[df.category == "English (prose)"]
    .set_index("tokenizer")["tokens_per_char"]
)
df["fragmentation_ratio_vs_english"] = df.apply(
    lambda r: r["tokens_per_char"] / baseline[r["tokenizer"]], axis=1
)

pd.set_option("display.width", 140)
print("\n=== Fragmentation table ===")
print(df.round(3).to_string(index=False))

df.to_csv("tokenizer_fragmentation_results.csv", index=False)
print("\nSaved: tokenizer_fragmentation_results.csv")

# %% ------------------------------------------------------------------
# 4. BAR CHART: FRAGMENTATION RATIO BY CATEGORY, PER TOKENIZER
# -----------------------------------------------------------------------
categories = list(SAMPLES.keys())
tok_names = list(TOKENIZERS.keys())
x = np.arange(len(categories))
width = 0.8 / len(tok_names)

fig, ax = plt.subplots(figsize=(11, 6))
for i, tok_name in enumerate(tok_names):
    vals = [
        df[(df.category == c) & (df.tokenizer == tok_name)]["tokens_per_char"].values[0]
        for c in categories
    ]
    ax.bar(x + i * width, vals, width, label=tok_name)

ax.set_xticks(x + width * (len(tok_names) - 1) / 2)
ax.set_xticklabels(categories, rotation=25, ha="right")
ax.set_ylabel("Tokens per character (lower = more efficient)")
ax.set_title("Tokenizer fragmentation by language / domain")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fragmentation_by_category.png", dpi=150)
plt.show()
print("Saved: fragmentation_by_category.png")

# Second chart: fragmentation RATIO vs English, easier to read as an
# "X times worse than English" plot.
fig2, ax2 = plt.subplots(figsize=(11, 6))
for i, tok_name in enumerate(tok_names):
    vals = [
        df[(df.category == c) & (df.tokenizer == tok_name)]["fragmentation_ratio_vs_english"].values[0]
        for c in categories
    ]
    ax2.bar(x + i * width, vals, width, label=tok_name)
ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xticks(x + width * (len(tok_names) - 1) / 2)
ax2.set_xticklabels(categories, rotation=25, ha="right")
ax2.set_ylabel("Fragmentation ratio vs. English prose (1.0 = parity)")
ax2.set_title("How many times more tokens per character vs. English")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fragmentation_ratio_vs_english.png", dpi=150)
plt.show()
print("Saved: fragmentation_ratio_vs_english.png")

# %% ------------------------------------------------------------------
# 5. COST / LATENCY IMPACT TABLE
# -----------------------------------------------------------------------
# Published $/1K INPUT tokens. Edit these constants to match whichever
# model/provider you actually bill against — the point of using a SINGLE
# reference price across tokenizers below is to isolate the pure
# tokenization-efficiency effect on cost, holding price-per-token fixed.
#
# GPT-4o: $2.50 / 1M input tokens = $0.0025 / 1K  (OpenAI, published Aug 2026)
REFERENCE_PRICE_PER_1K_TOKENS = 0.0025

# Optional: each tokenizer's own "native" model at its own real/typical
# price, so you can also see cost differences that combine tokenizer
# efficiency AND pricing differences between models. EDIT to your
# actual provider/rate — the Llama 3 number below is a rough, illustrative
# open-weight hosting price and varies a lot by host.
NATIVE_PRICE_PER_1K_TOKENS = {
    "GPT-4o (tiktoken/cl100k_base)": 0.0025,   # OpenAI GPT-4o input price
    "Llama 3 (128k BPE)": 0.0002,               # illustrative open-weight hosting price — EDIT ME
    "mT5 (multilingual SentencePiece)": 0.0025, # no public per-token API price; reference rate used
}

cost_rows = []
for category in categories:
    for tok_name in tok_names:
        sub = df[(df.category == category) & (df.tokenizer == tok_name)].iloc[0]
        n_tok = sub["n_tokens"]
        cost_rows.append({
            "category": category,
            "tokenizer": tok_name,
            "n_tokens": n_tok,
            "cost_usd_at_reference_price": n_tok / 1000 * REFERENCE_PRICE_PER_1K_TOKENS,
            "cost_usd_at_native_price": n_tok / 1000 * NATIVE_PRICE_PER_1K_TOKENS[tok_name],
            "fragmentation_ratio_vs_english": sub["fragmentation_ratio_vs_english"],
        })

cost_df = pd.DataFrame(cost_rows)
print("\n=== Cost impact table (reference price = $0.0025 / 1K tokens for all tokenizers) ===")
print(cost_df.round(5).to_string(index=False))
cost_df.to_csv("tokenizer_cost_impact.csv", index=False)
print("\nSaved: tokenizer_cost_impact.csv")

# %% ------------------------------------------------------------------
# 6. PLAIN-ENGLISH SUMMARY SENTENCES
# -----------------------------------------------------------------------
print("\n=== Summary ===")
non_english = [c for c in categories if c != "English (prose)"]
for tok_name in tok_names:
    print(f"\n[{tok_name}]")
    for c in non_english:
        ratio = df[(df.category == c) & (df.tokenizer == tok_name)]["fragmentation_ratio_vs_english"].values[0]
        direction = "more" if ratio > 1 else "fewer"
        print(f"  {c} uses {ratio:.2f}x {direction} tokens per character than English "
              f"(same tokenizer) -> ~{ratio:.2f}x the cost per equivalent piece of content.")
