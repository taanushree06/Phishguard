import re
import math
from urllib.parse import urlparse

# ── Constants ────────────────────────────────────────────────
SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".xyz", ".top", ".club", ".gq"}
BRAND_KEYWORDS  = {"sbi", "bank", "paypal", "google", "amazon",
                   "facebook", "apple", "netflix", "microsoft", "login", "verify"}
SENSITIVE_WORDS = {"login", "verify", "password", "update",
                   "secure", "account", "confirm"}


def char_entropy(s: str) -> float:
    """Shannon entropy — high value means more random/obfuscated hostname."""
    if not s:
        return 0.0
    freq = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in freq if p > 0)


def extract_features(url: str) -> list:
    """
    Returns a list of 19 numeric features for a given URL.
    This EXACT function must be used in both train_model.py and app.py.
    """
    try:
        parsed   = urlparse(url)
        hostname = parsed.netloc.lower()
        path     = parsed.path.lower()
        full     = url.lower()
    except Exception:
        return [0] * 19

    return [
        # ── Length features ──────────────────────────────────
        len(url),                                                    # 0  url_len
        len(hostname),                                               # 1  host_len
        len(path),                                                   # 2  path_len

        # ── Protocol ────────────────────────────────────────
        0 if parsed.scheme == "https" else 1,                       # 3  is_http

        # ── IP address used as host ──────────────────────────
        1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)
          else 0,                                                    # 4  ip_in_host

        # ── Special character counts ─────────────────────────
        url.count("."),                                              # 5  dot_count
        url.count("-"),                                              # 6  hyphen_count
        url.count("@"),                                              # 7  at_count
        url.count("?"),                                              # 8  question_count
        url.count("="),                                              # 9  equals_count
        url.count("&"),                                              # 10 ampersand_count
        url.count("//"),                                             # 11 double_slash_count
        len(re.findall(r"[~!#$%^*()_+]", url)),                     # 12 special_char_count

        # ── Subdomain depth ──────────────────────────────────
        max(hostname.count(".") - 1, 0),                            # 13 subdomain_depth

        # ── Digit ratio in hostname ──────────────────────────
        sum(c.isdigit() for c in hostname) / (len(hostname) + 1),  # 14 digit_ratio

        # ── Suspicious signals ───────────────────────────────
        1 if any(hostname.endswith(t) for t in SUSPICIOUS_TLDS)
          else 0,                                                    # 15 bad_tld
        1 if any(k in full for k in BRAND_KEYWORDS) else 0,        # 16 brand_keyword
        1 if any(w in path for w in SENSITIVE_WORDS) else 0,       # 17 sensitive_path

        # ── Entropy ──────────────────────────────────────────
        char_entropy(hostname),                                      # 18 hostname_entropy
    ]