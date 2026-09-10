"""Command-line entry points; dataset commands need no ML dependencies."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import load_config, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="able", description="ABLE training, data, and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-data", "sample"):
        p = commands.add_parser(name)
        p.add_argument("--data", default="datasets")
        if name == "sample":
            p.add_argument("--split", choices=("train", "validation", "test"), default="train")
            p.add_argument("--limit", type=int, default=1)
    for name in ("train-sft", "train-classifier", "train-ppo"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "train-classifier":
            p.add_argument("--task", choices=("persona", "gender_age", "politeness", "empathy"))
    p = commands.add_parser("generate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="datasets")
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--output", default="runs/predictions.jsonl")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p = commands.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--data", default="datasets")
    p.add_argument("--split", choices=("train", "validation", "test"), default="test")
    p.add_argument("--classifiers", help="JSON object mapping the four tasks to trained checkpoints")
    p.add_argument("--bertscore", action="store_true")
    p.add_argument("--bertscore-model", default="roberta-large")
    p.add_argument("--bertscore-num-layers", type=int, help="Encoder layer count for a local BERTScore model")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="runs/metrics.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-data":
            from .data import verify_dataset
            result = verify_dataset(args.data)
            print(json.dumps(result, indent=2))
            if result.get("errors"):
                raise SystemExit(1)
            return
        if args.command == "sample":
            from .data import iter_examples
            for example in iter_examples(args.data, args.split, args.limit):
                print(json.dumps(asdict(example), ensure_ascii=False))
            return
        if args.command in ("train-sft", "train-classifier", "train-ppo"):
            config = load_config(args.config)
            if args.command == "train-sft":
                from .training import train_sft
                result = train_sft(config)
            elif args.command == "train-classifier":
                from .training import train_classifier
                if args.task:
                    config["task"] = args.task
                    config["training"]["output_dir"] = "checkpoints/" + args.task
                result = train_classifier(config)
            else:
                from .ppo import train_ppo
                result = train_ppo(config)
        elif args.command == "generate":
            from .data import iter_examples
            from .inference import generate_dataset
            result = generate_dataset(args.checkpoint, iter_examples(args.data, args.split, args.limit),
                                      args.output, device=args.device, seed=args.seed,
                                      batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                      max_length=args.max_length, temperature=args.temperature, top_p=args.top_p)
        else:
            from .data import iter_examples
            from .evaluation import evaluate_predictions
            classifiers, similarity = None, None
            if args.classifiers:
                from pathlib import Path
                from .rewards import ClassifierScorer
                classifiers = ClassifierScorer(json.loads(Path(args.classifiers).read_text()), device=args.device)
            if args.bertscore:
                from .rewards import BertScoreSimilarity
                similarity = BertScoreSimilarity(args.bertscore_model, device=args.device,
                                                 num_layers=args.bertscore_num_layers)
            result = evaluate_predictions(iter_examples(args.data, args.split), args.predictions,
                                          classifiers=classifiers, similarity=similarity)
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"able: {exc}\n")
