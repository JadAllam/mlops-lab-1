"""Move the champion alias only if a candidate beats it by more than noise."""

import argparse
import math
import os
import sys

import mlflow

MODEL_NAME = "food11"
METRIC = "test_accuracy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--test-size", type=int, default=1096)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return parser.parse_args()


def accuracy_of(client, version):
    run = client.get_run(version.run_id)
    value = run.data.metrics.get(METRIC)
    if value is None:
        sys.exit(f"run {version.run_id} has no {METRIC} metric")
    return value


def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    candidate = client.get_model_version(MODEL_NAME, args.version)
    acc_candidate = accuracy_of(client, candidate)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        champion = None

    if champion is None:
        client.set_registered_model_alias(MODEL_NAME, "champion", args.version)
        print(f"no champion yet, version {args.version} promoted ({METRIC}={acc_candidate:.4f})")
        return

    acc_champion = accuracy_of(client, champion)
    se = math.sqrt(acc_champion * (1 - acc_champion) / args.test_size)
    threshold = acc_champion + 2 * se

    print(f"candidate v{args.version}  {METRIC}={acc_candidate:.4f}")
    print(f"champion  v{champion.version}  {METRIC}={acc_champion:.4f}")
    print(f"se={se:.4f}  needs >= {threshold:.4f}")

    if acc_candidate >= threshold:
        decision = "promoted"
        client.set_registered_model_alias(MODEL_NAME, "champion", args.version)
    else:
        decision = "rejected"
        client.set_registered_model_alias(MODEL_NAME, "challenger", args.version)

    client.set_model_version_tag(MODEL_NAME, args.version, "decision", decision)
    client.set_model_version_tag(MODEL_NAME, args.version, "candidate_accuracy", f"{acc_candidate:.4f}")
    client.set_model_version_tag(MODEL_NAME, args.version, "champion_accuracy", f"{acc_champion:.4f}")
    print(decision)


if __name__ == "__main__":
    main()
