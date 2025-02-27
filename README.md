# noCodeML # Machine Learning Model Trainer

## Overview
This project is a web-based application that allows users to upload CSV or Excel files, train selected machine learning models on their data, and receive accuracy metrics for the trained models. The website provides an easy-to-use and readable UI, making machine learning more accessible for users.

## Features
- Upload CSV or Excel datasets.
- Select from multiple machine learning models.
- Train models on uploaded data.
- View accuracy of trained models.
- Hosted on GitHub for accessibility.
- Streamlit-based UI for a seamless user experience.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `streamlit`
- **Hosting**: GitHub

## Project Structure
- `preprocessing.py` - Handles data preprocessing tasks.
- `model_training.py` - Contains logic for training ML models.
- `app.py` - The main Streamlit UI file.
- `requirements.txt` - Lists dependencies required for the project.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Enhancements
- Add a feature to download the trained model for further use.
- Expand support for additional machine learning models.
- Improve UI for better visualization of model performance.

## Outcome
This project significantly improved my machine learning skills. It provides an intuitive way for users to train models on their own datasets and evaluate performance easily.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to submit a pull request.


