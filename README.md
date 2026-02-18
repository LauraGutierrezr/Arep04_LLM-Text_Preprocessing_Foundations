# Lab 04 – LLM Text Preprocessing & Embeddings

This lab implements, step by step, the **text preprocessing and embedding generation pipeline** used in large language models (LLMs), following **Chapter 2 – Working with Text Data** from *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

The focus is not on training a full LLM, but on **understanding and experimenting** with the key stages that happen before the model: tokenization, sliding‑window sequence creation, and the use of embedding layers.

## Lab Objectives

By the end of this lab, the student should be able to:

- **Explain** why language models cannot operate directly on raw text and require a numerical representation.
- **Apply** a BPE tokenizer (tiktoken GPT‑2) to convert text into sequences of integer IDs.
- **Build** training datasets using sliding windows controlled by `max_length` and `stride`.
- **Implement** a simple PyTorch `DataLoader` to generate next‑token (input, target) pairs.
- **Use** an embedding layer (`torch.nn.Embedding`) to map IDs to dense vectors.
- **Reason** about why embeddings encode meaning and how they relate to neural network concepts.
- **Analyze** the effect of `max_length` and `stride` on the number of samples and context overlap.

## Project Structure

At the root of the repository you will find the following main files:

- `ch02.ipynb`  
  Original Chapter 2 notebook from Raschka’s book (reference code). It is not modified; it is used as a guide and source of the “core code”.

- `embeddings.ipynb`  
  Main lab notebook. It contains:
  - Loading of the text file `the-verdict.txt`.
  - Tokenization with `tiktoken` (GPT‑2 BPE) and creation of `tokens`.
  - A `create_dataset` function that generates `(input_ids, target_ids)` sequences using `max_length` and `stride`.
  - Construction of a PyTorch `TensorDataset` and `DataLoader`.
  - An embedding layer `torch.nn.Embedding` applied to the input sequences.
  - An experiment varying `max_length` and `stride` and counting the resulting samples.
  - Several **custom markdown cells** explaining the motivation of each step and explicitly answering why embeddings encode meaning.

- `the-verdict.txt`  
  Public‑domain text (*The Verdict* by Edith Wharton), used as the training / example corpus for the tokenization and embedding pipeline.

- `README.md`  
  This document, providing the overall description of the lab, its structure, environment, and conclusions.

## Technologies and Libraries

- **Python 3.9+** – Main programming language.
- **Jupyter Notebook** – Interactive environment to run and document the experiment.
- **PyTorch (`torch`)** – For `Tensor`, `TensorDataset`, `DataLoader`, and the `Embedding` layer.
- **tiktoken** – GPT‑2 compatible BPE tokenizer.
- **NumPy (`numpy`)** – Basic numerical operations (used lightly).

## Execution Environment

It is recommended to work inside a Python virtual environment within the lab directory.

### Create and Activate Virtual Environment

From the lab folder:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

Inside the virtual environment:

```bash
pip install torch tiktoken notebook ipykernel numpy
```

> Note: `embeddings.ipynb` also contains a `%pip install ...` cell that can be used directly from the notebook.

### Running the Notebooks

1. Start Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open and run, in the recommended order:
   - `ch02.ipynb` (optional, as book reference).
   - `embeddings.ipynb` (main lab notebook).

3. Verify that all cells run without errors and that token counts, sample counts, tensor shapes, and the `max_length`/`stride` experiment outputs are printed as expected.

## Notebook Overview

### `ch02.ipynb` – Book Reference Code

This notebook comes directly from the official repository of *Build a Large Language Model (From Scratch)*. It includes:

- Manual tokenization with regular expressions and vocabulary construction.
- Simple tokenizers `SimpleTokenizerV1` and `SimpleTokenizerV2`.
- Introduction to BPE and use of `tiktoken`.
- Construction of a sliding‑window `GPTDatasetV1`.
- Examples of token and positional embeddings.

In the context of this lab, it is used as **supporting material** and source for the “core code” that is then re‑implemented and summarized in `embeddings.ipynb`.

### `embeddings.ipynb` – Main Lab Notebook

This notebook contains the implementation and analysis that are submitted as part of the lab. Its main sections are:

1. **Introduction**  
   Explains the goal of the notebook: to show how we go from raw text to embeddings ready to be consumed by an LLM or agentic system.

2. **Loading the Text (`the-verdict.txt`)**  
   Reads the Edith Wharton text and prints its size and a short preview.

3. **Tokenization with `tiktoken`**  
   Uses `tiktoken.get_encoding("gpt2")` to encode the text into an integer sequence (`tokens`).  
   A markdown cell explains why tokenization is necessary for a neural network to operate on text.

4. **Dataset Creation with Sliding Window**  
   Defines the function:

   ```python
   def create_dataset(token, max_length, stride):
       ...
       return torch.tensor(input_ids), torch.tensor(target_ids)
   ```

   which generates `(input_ids, target_ids)` pairs following the next‑token prediction scheme.  
   The roles of `max_length` (context size) and `stride` (window shift) are explained.

5. **PyTorch DataLoader**  
   From these sequences, a `TensorDataset` and then a `DataLoader` are created, yielding `(inputs, targets)` batches ready to train an autoregressive model.

6. **Embedding Layer**  
   Defines:

   ```python
   embedding = torch.nn.Embedding(vocab_size, embedding_dim)
   embedded_tokens = embedding(input_ids)
   ```

   and prints the output shape.  
   A markdown cell discusses what an embedding is and why it can be seen as a trainable lookup table that projects discrete IDs into a continuous vector space.

7. **Experiment with `max_length` and `stride`**  
   Runs several scenarios varying `max_length` and `stride`, counting how many samples are generated in each case.  
   A markdown cell analyzes how overlap increases the number of examples and why this can act as a form of temporal “data augmentation”.

8. **Conclusion**  
   Summarizes the full pipeline (text → tokens → IDs → windows → embeddings) and discusses how these representations are the direct input to a Transformer and, by extension, to agentic systems that reason over text.

## Main Conclusions

Some key conclusions drawn from `embeddings.ipynb` are:

1. **Tokenization as a Bridge Between Language and Linear Algebra**  
   Models can only operate on tensors. Tokenization and ID mapping represent natural language as numerical sequences over which matrix multiplications and backpropagation can be applied.

2. **Sliding Windows and Context**  
   Using `max_length` and `stride` to extract windows from the text allows you to:
   - Control the context length seen by the model.
   - Generate multiple overlapping examples that capture smooth transitions between positions.
   - Balance number of samples vs. computational cost.  
   A smaller stride yields more examples but also more redundancy and higher cost.

3. **Why Embeddings Encode Meaning**  
   Embeddings are parameters (rows of a matrix) updated via gradient descent to minimize prediction error.  
   Words appearing in similar contexts receive similar updates, so their vectors tend to cluster in space.  
   Thus, **the geometry of the embedding space reflects semantic and syntactic regularities**.

4. **Relation to Neural Networks**  
   An embedding layer is mathematically equivalent to applying a weight matrix to a one‑hot vector:  
   it is a specialized, efficient linear layer.  
   These embeddings are then combined with attention layers and MLPs, forming the foundation of modern LLMs.

5. **Connection to Agentic Systems**  
   In an LLM‑based agent, all decisions, plans, and actions build on the input embeddings and how they are transformed through the model’s layers.  
   Understanding this pipeline helps reason about the capabilities and limitations of such agents.

## Possible Extensions

Some ideas for future work or lab extensions:

- Compare different `embedding_dim` configurations and observe the impact on memory and representational capacity.
- Visualize embeddings (e.g., with PCA or t‑SNE) for a small subset of the vocabulary.
- Add positional embeddings and simulate the full input to a Transformer block.
- Implement a small autoregressive model (for example, an MLP or a simple attention block) using the same `DataLoader`.

## Author

- **Valentina Gutiérrez** – Implementation of `embeddings.ipynb`, analysis, and lab documentation.

This project is developed in the context of a university course and is intended for academic use only
