# Tutorials (Work in Progress)
Personal project to revisit all types of neural networks fromPerceptron to Transformer and state space models
and some examples with visualizations. 

ToDo: Improve explaination and finish transformer, vae, state space model, pvrnn and mtrnn models. Also fix the links for the images

# Biological Neuron and Neural Network Simulation

## Overview
This document provides an explanation of the **Leaky Integrate-and-Fire (LIF) neuron model** and its extension to a **biological neural network simulation**.

## Leaky Integrate-and-Fire Neuron
The LIF neuron is a simplified mathematical model of biological neurons. It models how a neuron integrates input signals and fires an action potential when a threshold is reached.

### **Equations**
The membrane potential \( V(t) \) of the neuron evolves according to:

\[
\tau \frac{dV}{dt} = -(V - V_{rest}) + R I(t)
\]

where:
- \( \tau \) = Membrane time constant (ms)
- \( V_{rest} \) = Resting membrane potential (mV)
- \( R \) = Membrane resistance (MΩ)
- \( I(t) \) = Input current (nA)

If \( V \) reaches a threshold value, the neuron fires and resets to a lower potential.

### **Simulation & Visualization**
The neuron receives input current, and its membrane potential is plotted over time, visualizing its firing behavior.

**Figure 1: LIF Neuron Simulation**

![LIF Neuron](lif_neuron_plot.png)

## Biological Neural Network
A simple neural network is built using multiple LIF neurons, where neurons interact with each other through synaptic connections.

### **Network Dynamics**
Each neuron receives:
1. **External input currents**
2. **Influence from other neurons** via weighted connections

### **Equation with Synaptic Input**
\[
\tau \frac{dV_i}{dt} = -(V_i - V_{rest}) + R I_i + \sum_{j} W_{ij} V_j
\]

where \( W_{ij} \) is the connection strength between neurons.

### **Simulation & Visualization**
The network activity is animated, showing how neurons fire in response to stimuli.

**Figure 2: Neural Network Simulation**

![Neural Network](neural_network_plot.png)

## Conclusion
This simulation demonstrates how simple neuron models can be used to build biologically inspired neural networks. These models help us understand **information processing in the brain** and develop **biologically plausible AI systems**.

---

### **References**
- Dayan, P., & Abbott, L. (2001). Theoretical Neuroscience.
- Gerstner, W., & Kistler, W. (2002). Spiking Neuron Models.



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

