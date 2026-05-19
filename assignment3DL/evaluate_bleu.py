from __future__ import annotations
import argparse
from typing import List
import torch
from datasets import load_dataset
from tqdm import tqdm
from model import Transformer


def evaluate_bleu_infer(
    model:        Transformer,
    test_data,
    device:       torch.device,
    max_samples:  int = 1000,
) -> float:
    
    from evaluate import load as eval_load  
    bleu_metric = eval_load("bleu")

    model.eval()
    hypotheses: List[str]        = []
    references:  List[List[str]] = []

    samples = list(test_data)[:max_samples]

    for item in tqdm(samples, desc="Translating"):
        de_sentence = item["de"]
        en_reference = item["en"].lower()

        prediction = model.infer(de_sentence)

        hypotheses.append(prediction)
        references.append([en_reference])

    result = bleu_metric.compute(predictions=hypotheses, references=references)
    bleu   = result.get("bleu", 0.0)
    return bleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer BLEU on Multi30k test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_best.pt",
        help="Path to model weights (default: model_best.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Number of test sentences to evaluate (max 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[evaluate_bleu] Loading model on {device} …")
    model = Transformer(device=device)
    model.to(device)
    model.eval()

    print("[evaluate_bleu] Loading Multi30k test split …")
    raw       = load_dataset("bentrevett/multi30k")
    test_data = raw["test"]

    print(f"[evaluate_bleu] Translating {min(args.max_samples, len(test_data))} sentences …")
    bleu = evaluate_bleu_infer(model, test_data, device, max_samples=args.max_samples)

    print(f"\n{'=' * 40}")
    print(f"  Corpus BLEU : {bleu * 100:.2f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
