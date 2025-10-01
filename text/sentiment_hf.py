#!/usr/bin/env python
import argparse
from transformers import pipeline
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",
                        nargs="+",
                        required=True,
                        help="One or more texts"
                        )
    args = parser.parse_args()

    sys.stdout.flush()

    clf = pipeline("sentiment-analysis",
                   model=MODEL,
                   tokenizer=MODEL,
                   device=-1
                   )

    sys.stdout.flush()

    results = clf(args.text)

    for t, r in zip(args.text, results):
        print(f"TEXT: {t}\nLABEL: {r['label']} SCORE: {r['score']:.4f}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

# python sentiment_hf.py --text "Очень понравился фильм!"
