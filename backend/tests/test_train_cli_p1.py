"""Test CLI flags for P1."""
import sys
sys.path.insert(0, "..")

from train_cnn_bigru_multitask import parse_args


def test_augment_flag_exists():
    args = parse_args(["--augment"])
    assert args.augment is True
    print("PASS: test_augment_flag_exists")


def test_default_augment_is_false():
    args = parse_args([])
    assert args.augment is False
    print("PASS: test_default_augment_is_false")


if __name__ == "__main__":
    test_augment_flag_exists()
    test_default_augment_is_false()
