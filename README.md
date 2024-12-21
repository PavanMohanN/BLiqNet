![image](https://github.com/user-attachments/assets/52360cb2-f928-4f80-906f-d63393997274)

# BLiqNet: Bidirectional Liquid Neural Network (BLiqNet)

## Overview

BLiqNet (Bidirectional Liquid Neural Network) is a cutting-edge model designed for predicting and reconstructing concrete compressive strength. This model leverages the power of liquid neural networks, which combine the flexibility of recurrent neural networks with continuous-time differential equations. The bidirectional design enables the model to make predictions both forward (from input features to output) and backward (from output back to the original features).

This model is specifically trained to predict the compressive strength of concrete from various constituent materials and conditions, such as the amount of cement, water, and aggregates.

## Features

- **Bidirectional Neural Network**: BLiqNet allows for both forward and backward predictions. 
  - **Forward Prediction**: Predict the compressive strength of concrete based on input features.
  - **Backward Prediction**: Given the compressive strength, estimate the original mix proportions of concrete ingredients.
  
- **Liquid Neural Network Architecture**: The model uses a liquid neural network, which is more adaptive and robust to different types of data than conventional networks. It is particularly useful for problems where data can be modeled as continuous processes or sequences.

- **Customizable Inputs**: Users can provide various concrete ingredient parameters, and the model will output either the compressive strength or reconstruct the ingredient mix for a given strength.

## Requirements

Before running the code, ensure you have the necessary libraries installed:

- Python 3.x
- PyTorch
- scikit-learn
- NumPy
- Joblib
- Matplotlib

To install the required dependencies, use the following:

```bash
pip install torch scikit-learn numpy joblib matplotlib
```

## Model Description

BLiqNet consists of two main components:

1. **Liquid Neural Layer**: A core component of the liquid neural network that models a continuous-time process. It uses the `odeint_adjoint` solver to solve the differential equations of the system dynamically.
   
2. **Bidirectional Flow**: The model has two parts:
   - **Forward Prediction**: Takes the mix of concrete ingredients as input and predicts the compressive strength.
   - **Backward Prediction**: Given the compressive strength, the model reconstructs the ingredient mixture that could have produced that strength.

## How to Use

### Step 1: Load the Pretrained Model

You can use the pretrained model to make predictions. The model and its associated scalers must be loaded first.

```python
import torch
import joblib

# Load the entire model (including architecture and weights)
model = torch.load('path_to_your_model.pth')
model.eval()

# Load the scalers for input and output
scaler_X = joblib.load('path_to_your_scaler_X.pkl')
scaler_y = joblib.load('path_to_your_scaler_y.pkl')
```

### Step 2: Make a Prediction

#### Forward Prediction

To make a forward prediction, input the concrete mix proportions (e.g., cement, water, aggregates) and the model will output the compressive strength.

```python
import numpy as np

def get_input_for_forward():
    cement = float(input("Cement: "))
    blast_furnace_slag = float(input("Blast Furnace Slag: "))
    fly_ash = float(input("Fly Ash: "))
    water = float(input("Water: "))
    superplasticizer = float(input("Superplasticizer: "))
    coarse_aggregate = float(input("Coarse Aggregate: "))
    fine_aggregate = float(input("Fine Aggregate: "))
    age_day = float(input("Age (in days): "))
    
    return np.array([cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age_day]).reshape(1, -1)

# Get input from the user
user_input = get_input_for_forward()

# Scale the input
user_input_scaled = scaler_X.transform(user_input)
user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

# Make the forward prediction
with torch.no_grad():
    forward_pred, _ = model(user_input_tensor)

# Convert back to original scale
forward_pred = scaler_y.inverse_transform(forward_pred.cpu().numpy())

print(f"Forward Prediction (Concrete Compressive Strength): {forward_pred[0][0]:.4f}")
```

#### Backward Prediction

To make a backward prediction, input the compressive strength and the model will reconstruct the original concrete mix.

```python
def perform_backward_prediction():
    output_value = float(input("Please enter the forward prediction output value (Concrete Compressive Strength): "))
    output_value_scaled = scaler_y.transform(np.array([[output_value]]))  # Scale the output value
    output_value_tensor = torch.tensor(output_value_scaled, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():
        backward_pred = model.fc_backward(output_value_tensor)

    # Convert the backward prediction back to the original scale using scaler_X
    backward_pred = scaler_X.inverse_transform(backward_pred.cpu().numpy())

    print("Backward Prediction (Input Features):")
    print(f"Cement: {backward_pred[0][0]:.4f}")
    print(f"Blast Furnace Slag: {backward_pred[0][1]:.4f}")
    print(f"Fly Ash: {backward_pred[0][2]:.4f}")
    print(f"Water: {backward_pred[0][3]:.4f}")
    print(f"Superplasticizer: {backward_pred[0][4]:.4f}")
    print(f"Coarse Aggregate: {backward_pred[0][5]:.4f}")
    print(f"Fine Aggregate: {backward_pred[0][6]:.4f}")
    print(f"Age (days): {backward_pred[0][7]:.4f}")
```

### Step 3: Running the Program

When you run the script, the user is prompted to choose between **Forward Prediction** or **Backward Prediction**.

#### Forward Prediction

```
Choose prediction type:
1. Forward Prediction
2. Backward Prediction
Enter 1 for Forward Prediction or 2 for Backward Prediction: 1
Please provide the following input values:
Cement: 300
Blast Furnace Slag: 120
Fly Ash: 60
Water: 150
Superplasticizer: 5
Coarse Aggregate: 900
Fine Aggregate: 700
Age (in days): 28
Forward Prediction (Concrete Compressive Strength): 45.2634
```

#### Backward Prediction

```
Choose prediction type:
1. Forward Prediction
2. Backward Prediction
Enter 1 for Forward Prediction or 2 for Backward Prediction: 2
Please enter the forward prediction output value (Concrete Compressive Strength): 45.2634
Backward Prediction (Input Features):
Cement: 300.0000
Blast Furnace Slag: 120.0000
Fly Ash: 60.0000
Water: 150.0000
Superplasticizer: 5.0000
Coarse Aggregate: 900.0000
Fine Aggregate: 700.0000
Age (days): 28.0000
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Liquid Neural Networks**: The liquid neural network architecture was originally proposed to handle continuous data and is well-suited for tasks such as time-series prediction and modeling physical systems.
- **Bidirectional Neural Networks**: The bidirectional architecture allows the model to understand both the forward and backward dependencies, which is crucial for tasks like reconstructing the input features from the output.

# List of dependencies and their versions required to run BLiqNet are mentioned below

To create a new environment with **Python 3.10.12** in Anaconda and install the required libraries, follow these steps:

### Step 1: Create a New Environment with Python 3.10.12

You can create a new environment with Python 3.10.12 using the `conda create` command. Open a terminal and run the following command:

```bash
conda create --name myenv python=3.10.12
```

- This will create an environment named `myenv` with Python version 3.10.12.
- You can replace `myenv` with any name you prefer for the environment.

### Step 2: Activate the New Environment

Once the environment is created, activate it using:

```bash
conda activate myenv
```

### Step 3: Install Required Packages

Now that you're in the environment with Python 3.10.12, you can install the necessary libraries using `pip`. Run the following commands to install each package:

```bash
pip install torch==2.5.1+cu121
pip install torchdiffeq==0.2.5
pip install torch-optimizer==0.3.0
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scikit-learn==1.5.2
pip install matplotlib==3.8.0
pip install scipy==1.13.1
pip install h5py==3.12.1
pip install joblib==1.4.2
```

Alternatively, if you have a `requirements.txt` file, you can install all the dependencies at once using:

```bash
pip install -r requirements.txt
```

### Step 4: Verify the Installation

After installing the libraries, you can check the installed versions using:

```bash
pip show torch torchdiffeq torch-optimizer numpy pandas scikit-learn matplotlib scipy h5py joblib
```

This will print the installed version of each library.

### Step 5: Deactivate the Environment

Once you're done, you can deactivate the environment:

```bash
conda deactivate
```

### Summary of Commands:

1. **Create environment with Python 3.10.12**:
   ```bash
   conda create --name myenv python=3.10.12
   ```
2. **Activate the environment**:
   ```bash
   conda activate myenv
   ```
3. **Install libraries**:
   ```bash
   pip install torch==2.5.1+cu121
   pip install torchdiffeq==0.2.5
   pip install torch-optimizer==0.3.0
   pip install numpy==1.26.4
   pip install pandas==2.2.2
   pip install scikit-learn==1.5.2
   pip install matplotlib==3.8.0
   pip install scipy==1.13.1
   pip install h5py==3.12.1
   pip install joblib==1.4.2
   ```
4. **Verify installation**:
   ```bash
   pip show torch torchdiffeq torch-optimizer numpy pandas scikit-learn matplotlib scipy h5py joblib
   ```
5. **Deactivate the environment**:
   ```bash
   conda deactivate
   ```

This should set up your environment with Python 3.10.12 and the necessary libraries.
