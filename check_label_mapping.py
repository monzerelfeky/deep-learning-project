from src.data.dataset import load_cub_dataset


def main():

    train_dataset = load_cub_dataset("train")
    test_dataset = load_cub_dataset("test")

    print("\n========== LABEL MAPPING CHECK ==========")

    train_labels = train_dataset.features["label"]
    test_labels = test_dataset.features["label"]

    print("\nTraining label feature:")
    print(train_labels)

    print("\nTesting label feature:")
    print(test_labels)

    if train_labels == test_labels:
        print("\nPASSED: Label features match.")
    else:
        print("\nWARNING: Label features differ.")
        print("Manual mapping verification is required.")

    train_ids = set(train_dataset["id"])
    test_ids = set(test_dataset["id"])

    overlap = train_ids.intersection(test_ids)

    print("\nOverlapping image IDs:", len(overlap))

    if not overlap:
        print("PASSED: No overlapping image IDs.")
    else:
        print("WARNING: Overlapping image IDs detected.")

    print("\n========================================")


if __name__ == "__main__":
    main()