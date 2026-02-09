import random, os

# Paths
input_file = r"C:\Users\shahb\OneDrive\Desktop\GMB_dataset.txt"
output_dir = r"C:\Users\shahb\OneDrive\Desktop\ner_data"
os.makedirs(output_dir, exist_ok=True)

# Read dataset (use latin1 to handle Windows special chars)
with open(input_file, encoding="latin1") as f:
    content = f.read().strip()

# Split into sentences by blank lines
sentences = content.split("\n\n")

# Shuffle for randomness
random.shuffle(sentences)

# Split 80/10/10
n = len(sentences)
train_split = int(0.8 * n)
dev_split = int(0.9 * n)

train_sents = sentences[:train_split]
dev_sents = sentences[train_split:dev_split]
test_sents = sentences[dev_split:]

# Save files
with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_sents))
with open(os.path.join(output_dir, "dev.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(dev_sents))
with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(test_sents))

print("✅ Done splitting!")
print(f"Train: {len(train_sents)} sentences")
print(f"Dev:   {len(dev_sents)} sentences")
print(f"Test:  {len(test_sents)} sentences")
