from src.data.dataset import load_cub_dataset
from src.data.splitting import create_stratified_split


print("Loading original training dataset...")

dataset = load_cub_dataset("train")


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

train_ids = set(train_dataset["id"])
val_ids = set(val_dataset["id"])

assert train_ids.isdisjoint(val_ids), (
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

print("\nSplit indices saved successfully!")

print("\nALL TESTS PASSED!")