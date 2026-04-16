"""Test CLI argument defaults for train_cnn_bigru_multitask."""
import sys
sys.path.insert(0, "..")

from train_cnn_bigru_multitask import parse_args


def test_default_split_is_strict():
    args = parse_args([])
    assert args.split == "strict", f"Expected 'strict', got {args.split}"
    print("PASS: test_default_split_is_strict")


def test_default_lr_is_paper_value():
    args = parse_args([])
    assert args.lr == 1e-4, f"Expected 1e-4, got {args.lr}"
    print("PASS: test_default_lr_is_paper_value")


def test_use_lso_flag_exists():
    args = parse_args(["--use-lso"])
    assert args.use_lso is True
    print("PASS: test_use_lso_flag_exists")


if __name__ == "__main__":
    test_default_split_is_strict()
    test_default_lr_is_paper_value()
    test_use_lso_flag_exists()
