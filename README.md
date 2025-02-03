# Tutorials (Work in Progress)
Personal project to revisit all types of neural networks from to Transformer and state space models
and some examples with visualizations. 

ToDo: Improve explaination and finish transformer, vae, state space model, pvrnn and mtrnn models. Also fix the links for the images

# Neural Network Models Explained

This document explains the different neural network models implemented in `models.py` and how they work.

## 1. Convolutional Neural Network (CNN)

**Architecture:**
- Uses convolutional layers to extract spatial features from images.
- Pooling layers reduce dimensionality while preserving essential information.
- Fully connected layers classify the extracted features.

![CNN Architecture](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

## 2. Recurrent Neural Network (RNN)

**Architecture:**
- Processes sequences by maintaining a hidden state.
- Useful for tasks like time series prediction and natural language processing.



## 3. Long Short-Term Memory (LSTM)

**Architecture:**
- A specialized RNN that solves the vanishing gradient problem.
- Uses gates (input, forget, output) to regulate information flow.

![LSTM Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/The_LSTM_cell.png/640px-The_LSTM_cell.png)

## 4. Gated Recurrent Unit (GRU)

**Architecture:**
- A simplified version of LSTM with fewer parameters.
- Uses reset and update gates to control hidden states.



## 5. Artificial Neural Network (ANN)

**Architecture:**
- Consists of fully connected layers.
- Each neuron applies an activation function to transform input data.

![ANN Architecture](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

## 6. Transformer

**Architecture:**
- Uses self-attention mechanisms to weigh input sequences differently.
- Highly effective for NLP tasks like translation and text generation.


## 7. MAMBA Model

**Architecture:**
- Placeholder for a specialized model (e.g., memory-augmented or attention-based).
- Can be adapted based on specific requirements.

---
## **How to Use These Models**

1. Select a model in `main.py` when prompted.
2. Train the model on MNIST dataset.
3. Evaluate accuracy using test data.
4. Adjust hyperparameters via `config.json`.

