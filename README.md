# Oral Cancer Detection Project

## Project Overview
This project focuses on the detection of oral cancer using multimodal data, including histopathological images and photographic data. The repository contains preprocessing scripts, training scripts, and models for cancer classification.

## Folder Structure
- `archive/`: Contains the raw datasets for training, validation, and testing.
- `models/`: Includes the model architectures and saved models.
- `photo_split/`: Preprocessed photographic data split into train, validation, and test sets.
- `preprocessing/`: Scripts for preprocessing histopathological and photographic data.
- `train.py`: Script for training the models.
- `test.py`: Script for testing the models.
- `evaluate.py`: Script for evaluating the model performance.
- `utils.py`: Utility functions used across the project.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/ChanikyaSaiL/Cancer.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Cancer
   ```
3. Install the required dependencies (if applicable).
4. Run the preprocessing scripts to prepare the data.
5. Train the model using `train.py`.
6. Evaluate and test the model using `evaluate.py` and `test.py`.

## Acknowledgments
This project uses publicly available datasets for oral cancer detection. Special thanks to the contributors of these datasets.

## License
This project is licensed under the MIT License.