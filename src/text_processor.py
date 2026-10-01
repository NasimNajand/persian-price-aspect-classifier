import re
from typing import List
from hazm import stopwords_list, Normalizer, word_tokenize, Lemmatizer


class PersianTextProcessor:
    """Encapsulates Persian text normalization, regex pattern masking, and lemmatization."""

    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()

        # Domain-specific tokens preserved from stopword removal
        self.price_keep = {
            "قیمت", "قيمت", "تومان", "تومن", "ریال", "ريال",
            "ارزان", "ارزون", "ارزونتر", "ارزانتر",
            "گران", "گرون", "گرونتر", "گرانتر",
            "هزینه", "پول", "قیمتش", "قیمتها", "قیمتهاش"
        }
        self.stopwords = set(stopwords_list()) - self.price_keep

        self.re_punct = re.compile(r"[^\w\s‌]")
        self.re_multi_space = re.compile(r"\s+")
        self.re_v_prefix = re.compile(r"\bو(?=\S)")
        self.re_price_pair = re.compile(
            r"(\d[\d,\.]*)\s*(تومان|تومن|ريال|ریال|هزار|میلیون|k|K)"
        )
        self.re_number = re.compile(r"\d+([.,]\d+)*")

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = self.normalizer.normalize(text)
        text = self.re_v_prefix.sub("و ", text)
        text = self.re_price_pair.sub(r"PRICE_TOKEN \1 CUR_TOKEN \2", text)
        text = self.re_number.sub("NUM_TOKEN", text)
        text = self.re_punct.sub(" ", text)
        text = self.re_multi_space.sub(" ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)

        fixed_tokens = []
        for token in tokens:
            if token.startswith("و") and len(token) > 1 and token not in ("ول", "وای"):
                fixed_tokens.extend(["و", token[1:]])
            else:
                fixed_tokens.append(token)

        final_tokens = []
        for token in fixed_tokens:
            if token in self.stopwords and token not in self.price_keep:
                continue
            if token in ("NUM_TOKEN", "PRICE_TOKEN", "CUR_TOKEN"):
                final_tokens.append(token)
                continue
            final_tokens.append(self.lemmatizer.lemmatize(token))

        return final_tokens

    def preprocess_to_string(self, text: str) -> str:
        return " ".join(self.tokenize_and_lemmatize(text))