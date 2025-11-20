## 📝 Predicting Chaotic Systems with Machine Learning (Double Pendulum)

This repository contains the Jupyter Notebook **`LTSM-Double-Pendulum.ipynb`**, which explores the use of machine learning to model and predict the behaviour of a complex, chaotic dynamical system from physics.

***

### 🎯 Overview and Objective

The project focuses on the classic physics problem of the **double pendulum**. The notebook is structured around the following key tasks:

1.  **Physics Simulation:** We use **Lagrangian mechanics** to derive the equations of motion (EOMs) for the double pendulum  and numerically simulate its chaotic, non-linear dynamics using `scipy.integrate.solve_ivp`.
2.  **Data Generation:** High-fidelity time-series data for the pendulum's state variables (angles and angular velocities) is generated to train the model.
3.  **Machine Learning Task:** A **Recurrent Neural Network (RNN)**, specifically a **Long Short-Term Memory (LSTM)** model, is designed and trained to predict the future state (trajectory) of the system's masses.
4.  **Incomplete Information Challenge:** The core challenge is assessing the model's ability to predict the *full system state* (all four variables: $\theta_1$, $\dot{\theta}_1$, $\theta_2$, $\dot{\theta}_2$) when provided with only *partial* or incomplete input data, such as observing only the Cartesian coordinates of the lower mass.

The overall goal is to evaluate the efficacy of sequence prediction models in forecasting the complex, chaotic evolution of a physics system, particularly when faced with the inherent limitations of incomplete observational data.

***

### 🛠️ Dependencies

The notebook requires the following standard Python libraries:

* **`numpy`**: For numerical computations and array manipulation.
* **`matplotlib`**: For visualising the time-series data and pendulum trajectories.
* **`scipy.integrate`**: Specifically `solve_ivp` for numerically integrating the differential equations.
* **`tensorflow` / `keras`**: For building and training the LSTM prediction model.
