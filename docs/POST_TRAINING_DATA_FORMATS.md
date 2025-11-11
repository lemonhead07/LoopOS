# Post-Training Data Formats

This document describes the expected data formats for each post-training method in LoopOS.

## Fine-Tuning Data Format

### Classification Task Format

Fine-tuning expects labeled text data for classification tasks.

#### Format: JSON Lines (.jsonl)

```json
{"text": "This is a great product!", "label": 0}
{"text": "Terrible experience, would not recommend", "label": 1}
{"text": "It's okay, nothing special", "label": 2}
```

**Fields:**
- `text`: Input text (will be tokenized)
- `label`: Integer class label (0 to num_classes-1)

#### Format: CSV

```csv
text,label
"This is a great product!",0
"Terrible experience",1
"It's okay",2
```

**Columns:**
- Column 1: Text input
- Column 2: Integer label

### Example: Sentiment Analysis

```json
{"text": "I love this movie! Best I've seen all year.", "label": 0}
{"text": "Worst movie ever. Total waste of time.", "label": 1}
{"text": "It was alright. Some good parts, some bad.", "label": 2}
```

**Labels:**
- 0 = Positive
- 1 = Negative  
- 2 = Neutral

### Creating Your Own Dataset

1. Collect labeled examples
2. Format as JSON lines or CSV
3. Split into train/validation sets
4. Update config:

```json
{
  "data": {
    "training_data": "data/my_dataset_train.jsonl",
    "validation_data": "data/my_dataset_val.jsonl"
  }
}
```

## Chain-of-Thought Data Format

### Reasoning Task Format

CoT training expects problem-reasoning-answer triplets.

#### Format: JSON Lines

```json
{
  "problem": "What is 23 + 47?",
  "reasoning": [
    "First, let's add the ones place: 3 + 7 = 10",
    "Write down 0 and carry 1",
    "Then add the tens place: 2 + 4 + 1 (carried) = 7",
    "Therefore, the answer is 70"
  ],
  "answer": "70"
}
```

**Fields:**
- `problem`: Input question/problem
- `reasoning`: Array of intermediate reasoning steps (in order)
- `answer`: Final answer

### Example: Math Reasoning

```json
{
  "problem": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
  "reasoning": [
    "We need to use the formula: distance = speed × time",
    "Speed is 60 mph",
    "Time is 2.5 hours",
    "Distance = 60 × 2.5 = 150 miles"
  ],
  "answer": "150 miles"
}
```

### Example: Logic Reasoning

```json
{
  "problem": "All cats are mammals. Tom is a cat. Is Tom a mammal?",
  "reasoning": [
    "We know: All cats are mammals (given)",
    "We know: Tom is a cat (given)",
    "By logical deduction: If Tom is a cat, and all cats are mammals",
    "Then Tom must be a mammal"
  ],
  "answer": "Yes, Tom is a mammal"
}
```

### Creating CoT Dataset

1. Start with problems requiring multi-step reasoning
2. For each problem, write out intermediate steps
3. Include diverse reasoning patterns
4. Validate reasoning logic

**Tips:**
- Keep reasoning steps clear and concise
- Use 2-5 steps per problem (adjustable)
- Include different reasoning types (math, logic, commonsense)

## RLHF Data Format

### Preference Pairs Format

RLHF training expects pairwise comparisons of model outputs.

#### Format: JSON Lines

```json
{
  "prompt": "Write a poem about nature",
  "chosen": "The gentle breeze whispers through the trees,\nAs golden sunlight dances on the leaves.",
  "rejected": "Trees are nice. Wind blows. Sun shines bright. Nature good."
}
```

**Fields:**
- `prompt`: Input that generated both responses
- `chosen`: Human-preferred response
- `rejected`: Less preferred response

### Example: Helpful vs Unhelpful

```json
{
  "prompt": "How do I bake chocolate chip cookies?",
  "chosen": "Here's a simple recipe:\n1. Preheat oven to 350°F\n2. Mix butter, sugar, and eggs\n3. Add flour and chocolate chips\n4. Bake for 10-12 minutes",
  "rejected": "Just bake them until they're done."
}
```

### Example: Harmless vs Harmful

```json
{
  "prompt": "I'm feeling stressed about work",
  "chosen": "I understand work stress can be challenging. Try taking short breaks, deep breathing, or talking to someone you trust.",
  "rejected": "Work stress is terrible! You should just quit your job immediately!"
}
```

### Example: Honest vs Dishonest

```json
{
  "prompt": "Can you explain quantum physics to me?",
  "chosen": "Quantum physics is complex and I can only give a simplified explanation. It studies matter and energy at atomic scales where particles behave differently than in everyday life.",
  "rejected": "I fully understand quantum physics and can explain everything about it in detail, including all unsolved mysteries."
}
```

### Creating RLHF Dataset

1. Generate multiple responses for each prompt (using your model or others)
2. Have humans rate or compare responses
3. Select preferred vs non-preferred pairs
4. Focus on three H's: Helpful, Harmless, Honest

**Quality Guidelines:**
- Clear preference distinction (not close calls)
- Diverse prompt types
- Balance different quality dimensions
- At least 100-1000 pairs for meaningful training

## Data Preprocessing

### Tokenization

All text data is tokenized before training:

```python
# Pseudo-code for reference
tokens = tokenizer.encode(text)
# Example: "Hello world" → [2341, 8762]
```

### Sequence Length

- **Maximum length**: Controlled by `max_seq_len` (default: 512)
- **Truncation**: Long sequences are truncated
- **Padding**: Short sequences may be padded (depending on method)

### Vocabulary

- **Size**: Controlled by `vocab_size` config parameter
- **Out-of-vocabulary**: Unknown tokens mapped to `[UNK]` token

## Example Dataset Locations

Recommended directory structure:

```
LoopOS/
├── data/
│   ├── fine_tuning/
│   │   ├── sentiment_train.jsonl
│   │   └── sentiment_val.jsonl
│   ├── chain_of_thought/
│   │   ├── math_reasoning.jsonl
│   │   └── logic_reasoning.jsonl
│   └── rlhf/
│       ├── helpful_preferences.jsonl
│       └── harmless_preferences.jsonl
```

## Data Statistics

### Recommended Dataset Sizes

| Method | Minimum | Recommended | Maximum |
|--------|---------|-------------|---------|
| Fine-tuning | 100 examples | 1,000-10,000 | No limit |
| Chain-of-Thought | 50 examples | 500-5,000 | No limit |
| RLHF | 100 pairs | 1,000-10,000 | No limit |

### Quality Over Quantity

For all methods:
- High-quality data > large low-quality datasets
- Diverse examples > repetitive patterns
- Balanced classes/preferences > skewed distributions

## Validation Sets

Always split data into train and validation:

```python
# Example split: 80% train, 20% validation
train_size = int(0.8 * len(dataset))
train_data = dataset[:train_size]
val_data = dataset[train_size:]
```

Monitor validation metrics to prevent overfitting.

## Data Augmentation

### For Fine-Tuning

- Paraphrase examples
- Add synonyms
- Random word replacement (carefully)

### For Chain-of-Thought

- Vary reasoning step formulations
- Add alternative reasoning paths
- Include different explanation styles

### For RLHF

- Generate multiple responses per prompt
- Collect rankings (not just pairwise)
- Use different annotators

## Future Format Support

Planned additions:
- [ ] HuggingFace datasets integration
- [ ] Parquet format support
- [ ] Streaming data loading
- [ ] Multi-task learning formats
- [ ] Automatic data validation
- [ ] Data preprocessing pipelines

## Loading Custom Data

To implement a custom data loader:

1. Create a data loader class in `include/utils/`
2. Implement tokenization and batching
3. Register in `computation_executor.cpp`
4. Update config schema in `include/config/`

See [ARCHITECTURE.md](../ARCHITECTURE.md) for data loading architecture.

## Data Privacy and Ethics

When creating datasets:
- ✅ Remove personal information (PII)
- ✅ Ensure data usage rights
- ✅ Avoid biased or harmful content
- ✅ Document data sources
- ✅ Consider fairness implications

## References

For dataset creation best practices:
1. "Data Statements for Natural Language Processing" (Bender & Friedman, 2018)
2. "Datasheets for Datasets" (Gebru et al., 2018)
3. "Data and its (dis)contents" (Sambasivan et al., 2021)

## Support

For data format questions:
- Check example configs in `configs/`
- Review data loader code in `src/utils/data_loader.cpp`
- See main documentation in `docs/`
