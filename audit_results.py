import ast
import io
import json
from pathlib import Path
import tokenize
import torch
from models import get_model


def main():
    results = []
    epochs = 0
    for path in sorted(Path("results").glob("*_seed*.json")):
        record = json.loads(path.read_text())
        config = record["config"]
        history = record["history"]
        assert len(history) == config["epochs"] or config["early_stopping"]
        expected_train = 50000 if config["dataset"] == "cifar10" else 100000
        assert all(row["train"]["samples"] == expected_train for row in history)
        assert all(row["validation"]["samples"] == 10000 for row in history)
        assert record["selected_train"]["samples"] == expected_train
        assert record["selected_validation"]["samples"] == 10000
        selected = history[record["best_epoch"] - 1]["validation"]
        assert abs(selected["accuracy"] - record["selected_validation"]["accuracy"]) < 0.0001
        assert abs(selected["macro_f1"] - record["selected_validation"]["macro_f1"]) < 0.0001
        expected_gap = record["selected_train"]["accuracy"] - selected["accuracy"]
        assert abs(expected_gap - record["generalization_gap"]) < 0.0001
        weights = torch.load(config["save_model"], map_location="cpu", weights_only=True)
        model = get_model(config["model"], num_classes=len(record["classes"]), norm_type=config["norm"])
        model.load_state_dict(weights, strict=True)
        assert all(torch.isfinite(value).all() for value in weights.values() if value.is_floating_point())
        assert Path(config["plot_file"]).is_file()
        results.append(path.stem)
        epochs += len(history)
    assert len(results) == 54, len(results)
    for path in [*Path("models").glob("*.py"), *Path("utils").glob("*.py"),
                 *Path("configs").glob("*.py"), *Path(".").glob("*.py")]:
        text = path.read_text()
        tree = ast.parse(text)
        assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(text).readline)), path
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                assert not (isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)), path
    summary = {"training_runs": len(results), "epochs": epochs,
               "selected_checkpoint_metrics_match": True, "all_training_samples_used": True,
               "finite_weights": True, "fresh_model_reload": True, "plots_present": True, "code_comments": 0}
    Path("results/audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
