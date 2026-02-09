from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import RunInfo, load_model, new_run_id, now_iso, save_run
from .features import FeatureConfig
from .inference import predict_next_day_direction
from .io import LoadConfig, load_frame
from .models import ModelConfig
from .splits import SplitConfig
from .train import TrainConfig, fit_final_model, walk_forward_cv


def main() -> None:
    parser = argparse.ArgumentParser(prog="fma-ml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--dataset-id", default=None)
    train.add_argument("--path", default=None)
    train.add_argument("--symbol", required=True)
    train.add_argument("--model", choices=["logreg", "dummy"], default="logreg")
    train.add_argument("--splits", type=int, default=5)
    train.add_argument("--out", default="artifacts")

    predict = sub.add_parser("predict")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--artifacts-dir", default="artifacts")
    predict.add_argument("--dataset-id", default=None)
    predict.add_argument("--path", default=None)
    predict.add_argument("--symbol", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        df = load_frame(dataset_id=args.dataset_id, path=args.path, cfg=LoadConfig())
        cfg = TrainConfig(
            feature=FeatureConfig(),
            split=SplitConfig(n_splits=int(args.splits)),
            model=ModelConfig(kind=str(args.model)),
        )

        cv = walk_forward_cv(df, symbol=str(args.symbol), cfg=cfg)
        model, n_rows, n_features = fit_final_model(df, symbol=str(args.symbol), cfg=cfg)

        info = RunInfo(
            run_id=new_run_id(),
            created_at=now_iso(),
            symbol=str(args.symbol),
            model_kind=str(args.model),
            n_rows=n_rows,
            n_features=n_features,
            metrics=cv,
        )
        out_path = save_run(output_dir=Path(args.out), model=model, info=info)
        print(str(out_path))
        return

    if args.cmd == "predict":
        df = load_frame(dataset_id=args.dataset_id, path=args.path, cfg=LoadConfig())
        model = load_model(artifacts_dir=Path(args.artifacts_dir), run_id=str(args.run_id))
        res = predict_next_day_direction(df, symbol=str(args.symbol), model=model, feature_cfg=FeatureConfig())
        print(f"{res.as_of},{res.symbol},{res.prob_up:.6f},{res.predicted_label}")
        return


if __name__ == "__main__":
    main()

