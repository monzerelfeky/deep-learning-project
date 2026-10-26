from src.data.dataset import load_cub_dataset

from src.data.captions import (
    clean_captions,
    preprocess_captions,
    validate_captions
)


def main():

    print("Loading CUB training dataset...")

    dataset = load_cub_dataset("train")

    print("\nChecking first image captions...")

    sample = dataset[0]

    description = sample["description"]

    captions = preprocess_captions(description)

    print("\nImage:", sample["file_name"])

    print("Number of captions:", len(captions))

    print("\nCleaned Captions:")

    for i, caption in enumerate(captions, start=1):
        print(f"{i}. {caption}")

    print("\nCaption validation:")

    if validate_captions(captions):
        print("PASSED: Found 10 valid captions.")
    else:
        print("WARNING: Unexpected caption structure.")

    print("\nChecking all training images...")

    valid_count = 0
    invalid_count = 0
    raw_total = 0
    cleaned_counts = []

    # Text column only: iterating over the dataset would decode every image
    for description in dataset["description"]:

        captions = preprocess_captions(description)
        raw_total += len(captions)

        if validate_captions(captions):
            valid_count += 1
        else:
            invalid_count += 1

        cleaned_counts.append(len(clean_captions(description)))

    print("\n========== RESULTS ==========")

    print("Total images:", len(dataset))
    print("Valid caption records:", valid_count)
    print("Invalid caption records:", invalid_count)

    print("\nRaw captions:", raw_total)
    print("Cleaned captions kept:", sum(cleaned_counts))
    print("Dropped (under 4 tokens or repeated):", raw_total - sum(cleaned_counts))
    print(
        "Cleaned captions per image:",
        min(cleaned_counts), "to", max(cleaned_counts)
    )

    print("\nCaption preprocessing check completed!")


if __name__ == "__main__":
    main()