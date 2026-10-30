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

    # id is only the row number inside each split, so every test id
    # also appears in train; file_name identifies the image
    train_names = set(train_dataset["file_name"])
    test_names = set(test_dataset["file_name"])

    overlap = train_names.intersection(test_names)

    print("\nFile names in both train and test:", len(overlap))

    if not overlap:
        print("PASSED: No overlapping image file names.")
    else:
        print("WARNING: Overlapping image file names detected.")

    print("\n========================================")


if __name__ == "__main__":
    main()
