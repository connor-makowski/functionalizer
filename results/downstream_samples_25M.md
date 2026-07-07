# Downstream Generation Samples Report (25M)

This report showcases a comparison of generated outputs between the different tokenizer configurations.

## Dataset: TinyStories (Natural Language Generation)

### Configuration: Split Only

#### Success Sample
* **Prompt**: `'Spot. Spot saw the shiny car and said, "Wow,'`
* **Generated Tokens (first 20)**: `['that', 's', 'a', 'nice', 'car', ' ', '!"', 'Spot', 'barked', 'and', 'barked', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'thatsanicecar !"Spotbarkedandbarked    '`
* **Completion Length (chars)**: 39

#### Failure Sample
* **Prompt**: `'Once upon a time, there was a big, strong robot'`
* **Generated Tokens (first 20)**: `[' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'    '`
* **Completion Length (chars)**: 4

---

### Configuration: Split + Functionalizer

#### Success Sample
* **Prompt**: `'Spot. Spot saw the shiny car and said, "Wow,'`
* **Generated Tokens (first 20)**: `['\ue100\ue000', ',', '\ue100\ue000', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `',    '`
* **Completion Length (chars)**: 5

#### Failure Sample
* **Prompt**: `'Once upon a time, there was a big, strong robot'`
* **Generated Tokens (first 20)**: `[' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'    '`
* **Completion Length (chars)**: 4

---

### Configuration: Split + Functionalizer (Repeat)

#### Success Sample
* **Prompt**: `'Spot. Spot saw the shiny car and said, "Wow,'`
* **Generated Tokens (first 20)**: `['that', 's', 'a', 'big', 'truck', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'thatsabigtruck    '`
* **Completion Length (chars)**: 18

#### Failure Sample
* **Prompt**: `'Once upon a time, there was a big, strong robot'`
* **Generated Tokens (first 20)**: `[' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'    '`
* **Completion Length (chars)**: 4

---

## Dataset: CSN-Python (Code Generation)

### Configuration: Split Only

#### Success Sample
*No success sample found for this configuration.*

#### Failure Sample
* **Prompt**: `'def save_act(self, path=None):'`
* **Generated Tokens (first 20)**: `[' ', '#', 'pragma', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `' #pragma    '`
* **Completion Length (chars)**: 12

---

### Configuration: Split + Functionalizer

#### Success Sample
* **Prompt**: `'def get_network_builder(name):'`
* **Generated Tokens (first 20)**: `['\ue200\ue000\ue004', '\ue200\ue000\ue003', '\n', ' ', 'returns', 'a', 'list', 'of', '\ue100\ue000', '\ue100\ue002', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'\n\n\n\n\n\n returnsalistof    '`
* **Completion Length (chars)**: 25

#### Failure Sample
* **Prompt**: `'def save_act(self, path=None):'`
* **Generated Tokens (first 20)**: `['\ue200\ue000\ue008', '\ue200\ue000\ue003', '\n', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'\n\n\n\n\n\n\n\n\n\n    '`
* **Completion Length (chars)**: 14

---

### Configuration: Split + Functionalizer (Repeat)

#### Success Sample
*No success sample found for this configuration.*

#### Failure Sample
* **Prompt**: `'def save_act(self, path=None):'`
* **Generated Tokens (first 20)**: `['\ue200\ue000\ue008', '\ue200\ue000\ue003', '\n', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'\n\n\n\n\n\n\n\n\n\n    '`
* **Completion Length (chars)**: 14

---

## Dataset: CSN-Go (Code Generation)

### Configuration: Split Only

#### Success Sample
*No success sample found for this configuration.*

#### Failure Sample
* **Prompt**: `'func (q *query) Close() {'`
* **Generated Tokens (first 20)**: `['\t', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'\t    '`
* **Completion Length (chars)**: 5

---

### Configuration: Split + Functionalizer

#### Success Sample
*No success sample found for this configuration.*

#### Failure Sample
* **Prompt**: `'func (q *query) Close() {'`
* **Generated Tokens (first 20)**: `['\t', '.', '.', '.', '.']`
* **Decoded Completion**: `'\t....'`
* **Completion Length (chars)**: 5

---

### Configuration: Split + Functionalizer (Repeat)

#### Success Sample
*No success sample found for this configuration.*

#### Failure Sample
* **Prompt**: `'func (q *query) Close() {'`
* **Generated Tokens (first 20)**: `['\t', ' ', ' ', ' ', ' ']`
* **Decoded Completion**: `'\t    '`
* **Completion Length (chars)**: 5

---

