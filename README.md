![image](https://github.com/user-attachments/assets/52360cb2-f928-4f80-906f-d63393997274)

# BLiqNet: Bidirectional Liquid Neural Network

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
