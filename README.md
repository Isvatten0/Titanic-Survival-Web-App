# Titanic Survival Prediction Web App

Welcome to the Titanic Survival Prediction Web App! This repository contains the code for a web application predicting the likelihood of survival for Titanic passengers based on various characteristics. The app, built using the Streamlit framework, integrates a pre-trained machine learning model. Here's an updated guide to facilitate a better understanding and usage of the application.

## Usage

1. **Run the App Locally:**
   - Execute the command `streamlit run app.py` to run the app locally.
   - Go to https://cs451app.streamlit.app/ to run the app on the web.

2. **Install Required Libraries (If Runnning Locally):**
   - Ensure you have the required libraries installed by running:
     ```bash
     pip install -r requirements.txt
     ```
     Alternatively, you can manually install the libraries:
     ```bash
     pip install numpy>=1.20.0 scikit-learn>=0.24.0 pandas==1.0.1 matplotlib==3.1.3 joblib==0.14.1 tpot==0.11.1 streamlit==0.61.0
     ```

3. **Python Version:**
   - The code is designed to run in Python 3.10.4. Ensure you have the appropriate Python version installed.

4. **User Input Section:**
   - Utilize the sidebar to input specific passenger characteristics: sex, passenger class, age, total siblings and spouses aboard, total parents and children aboard, fare, embarked port, and passenger title.

5. **Confidence Level Visualization:**
   - Explore the horizontal bar chart displaying the confidence level of the prediction.
   - The x-axis represents the percentage, and the y-axis indicates the predicted survival outcome.

6. **Prediction Outcome:**
   - Receive an outcome statement indicating the predicted survival status for the passenger.

7. **Definitions and How to Use Section:**
   - Gain insights into the definitions of features used in the prediction.
   - Follow instructions on effectively using the app to explore predictions.

## App Overview

### 1. User Input Section

- **Sidebar Interaction:**
  - Use dropdowns and sliders for an intuitive user experience.

- **Input Characteristics:**
  - Choose passenger characteristics such as sex, passenger class, age, family size, fare, embarked port, and passenger title.

### 2. Confidence Level Visualization

- **Bar Chart:**
  - Explore a horizontal bar chart showcasing the confidence level of the prediction.
  - Understand the prediction certainty with percentage values on the x-axis.

  **Example:**
  - Adjust the age slider to increase the passenger's age.
  - Observe how the confidence of survival may decrease, illustrating the impact of age on the model's prediction.

### 3. Prediction Outcome

- **Outcome Statement:**
  - Receive a clear outcome statement indicating the model's prediction for the passenger's survival status.

### 4. Definitions and How to Use Section

- **Feature Definitions:**
  - Understand the meanings of the features used in the prediction, enhancing user comprehension.

- **Usage Instructions:**
  - Follow a step-by-step guide on how to effectively use the app to explore predictions.


## App.py Overview

`app.py` serves as the main script for the Titanic Survival Prediction Web App. It incorporates the Streamlit framework to create an interactive and user-friendly interface. The script leverages a pre-trained machine learning model to predict the likelihood of survival for Titanic passengers based on input characteristics. The code facilitates a seamless user experience by integrating input controls, visualization components, and prediction outcomes.

### Key Components

1. **User Input Section:**
   - The sidebar allows users to input specific passenger characteristics using dropdowns and sliders.
   - Input features include sex, passenger class, age, total siblings and spouses aboard, total parents and children aboard, fare, embarked port, and passenger title.

2. **Confidence Level Visualization:**
   - Utilizes a horizontal bar chart to display the confidence level of the prediction.
   - The x-axis represents the percentage, while the y-axis indicates the predicted survival outcome.

3. **Prediction Outcome:**
   - Generates a clear outcome statement indicating the predicted survival status for the passenger.

4. **Definitions and How to Use Section:**
   - Provides insights into the definitions of features used in the prediction.
   - Offers instructions on effectively using the app to explore predictions.

## NotebookTitanic Overview

`NotebookTitanic.ipynb` is a Jupyter Notebook containing the data exploration, preprocessing, and machine learning model training processes. This notebook serves as the foundation for the model used in `app.py`. The documented code and analysis within the notebook contribute to the transparency and reproducibility of the project.

### Key Sections

1. **Data Exploration:**
   - Analyzes the Titanic dataset, exploring variable distributions and relationships.
   - Introduces the creation of additional variables, such as passenger title.

2. **Data Preprocessing:**
   - Addresses missing values and enhances variable interpretability through cleaning procedures.

3. **Model Training:**
   - Utilizes machine learning techniques, including TPOT and Logistic Regression, to optimize predictive accuracy.
   - Incorporates the refined Titanic dataset for training.

4. **Serialization:**
   - Follows the serialization process for efficient storage and deployment readiness.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or create a pull request.
