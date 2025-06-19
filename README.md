# Neuromorphic-Computing - Hardware for Efficient AI 

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 application using hardware-aware training based on real-world device measurements. The code simulates synapse behaviour with non-idealities such as non-linearity, asymmetry, and stochastic gain.

## Project Structure
<pre>
  Project Folder
      ├── Model.py              # Main training script 
      ├── Utils.py              # Utility functions for further analysis
      ├── DataAnalysisV2.ipynb  # Jupyter notebook for analysis of neasurement data
      ├── Test2.zip             # Dataset used
      ├── requirements.txt
      ├── README.md
</pre>

After running the scipts, the following folders will be created in the working directory
<pre>
  Project Folder
      ├── Images/               # Saved training plots
      ├── Videos/               # Recorded videos of trained agents
      ├── Best Models/          # Saved .pt files of best models 
      └── logs/                 # CSV logs of training
</pre>


Experimental files:
 - train.py: Attempt to consider drift
 - Data_Analysis.ipynb: For analysis of measurement data
 - Test.zip: Initial measurement data


## How to run

1. Install dependencies:
<pre>
  pip install -r requirements.txt
</pre>

2. Run training:
<pre>
  python Model.py
</pre>

3. Evaluation
<pre>
  python Utils.py
  python evaluate.py
</pre>

