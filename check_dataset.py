from src.data.dataset import load_cub_dataset, load_data_config
from src.data.splitting import (
    apply_exclusions,
    create_stratified_split,
    load_split_indices
)


print("Loading original training dataset...")

dataset = load_cub_dataset("train")


# Re-creates the saved split; fails instead of overwriting
# if the result differs from configs/split_indices.json
print("\nCreating stratified split...")

train_dataset, val_dataset = create_stratified_split(
    dataset,
    validation_size=0.10,
    random_seed=42
)


print("\n========== RESULTS ==========")

print("Original training images:", len(dataset))
print("New training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("\nChecking split integrity...")

assert len(train_dataset) + len(val_dataset) == len(dataset)

# id is only the row number, so file_name is the real image key
train_names = set(train_dataset["file_name"])
val_names = set(val_dataset["file_name"])

assert train_names.isdisjoint(val_names), (
    "ERROR: Training and validation overlap!"
)

print("No overlap between training and validation!")

print("\nChecking class coverage...")

train_classes = set(train_dataset["label"])
val_classes = set(val_dataset["label"])

assert train_classes == val_classes, (
    "ERROR: Class coverage differs between splits!"
)

print("Both splits contain the same classes.")

print("\nApplying exclusions from configs/data.yaml...")

train_indices, val_indices = load_split_indices(dataset)

leakage = [
    entry["file_name"]
    for entry in load_data_config()["exclusions"]["train_leakage"]
]

train_indices, val_indices, excluded = apply_exclusions(
    dataset, train_indices, val_indices, leakage
)

print("Training images used:", len(train_indices))
print("Validation images used:", len(val_indices))
print("Excluded images:", len(excluded))

assert len(train_indices) + len(val_indices) + len(excluded) == len(dataset)

print("\nSplit indices saved successfully!")

print("\nALL TESTS PASSED!")
