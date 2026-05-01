# Facial Emotion Recognition with dataset preprocessing 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)

This project implements a dataset preprocessing, model training and evaluation in a stratified cross-validation setup with experiment tracking pipeline for Facial Emotion Recognition task. The project was built with Pytorch. 

## Project description

The preprocessing steps include: 
* Setting up the FER+ dataset based on the original .csv files.
* Detecting duplicate images within the dataset and removing them.
* Setting up a stratified Cross Validation directories with train/validation/test splits.

Detecting duplicate images is done using computing phash and applying nearest-neighbor search which uses hamming distance as similarity metric. The duplicates are saved in seperate directories for visual verification. During the verification it was found that some of the same duplicate images were assigned to different classes, despite being corrected from FER+ 10-man voting. The total sum of votes for the duplicates were used to classify them to one class. The duplicates are removed keeping only one image sample.  

The training process includes: 
* Fine-tuning 3 choosen neural nets - ShuffleNetV2, ConveNext and Swin Transformer - for each iteration of stratified 5-fold Cross Validation. 

Metrics used for model evaluations: 
* Recall, Precision and F1 score for each emotion class with averaged values 
* Confusion matrixes
* Accuracy 

All the metrics are gathered from each iteration of Cross-Validation and then used to calculate mean values with standard deviations for the entire process. 
These metrics are plotted using matplotlib and the experiments results are logged via Weights & Biases. 

The .csv files used in this project can be found here:  
https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition
https://github.com/microsoft/ferplus 

--- 

## Tech Stack

- **Core**: Python, PyTorch  
- **Machine Learning**: scikit-learn
- **Computer Vision**: imagehash 
- **Visualization**: matplotlib, seaborn 
- **Experiment Tracking**: Weights & Biases  

---

## Project Structure

```text
UNet/
├── datasets/                           # Here are stored dirs for original dataset, k-fold CV subsets and duplicates found
   └── ...
├── metrics/           
    ├── classification_report.py        # Calculates classification report and plots it
    └── confusion_matrixes.py           # Calculates confusion matrix and plots
├── preprocessing/  
    ├── cv_fold_generator.py            # Creates structured directories for k-fold CV 
    ├── duplicate_finder.py             # Finds duplicates in original dataset
    ├── duplicate_remover.py            # Removes duplicates from original dataset
    └── fer_csv_reader.py               # Creates the directiory with original dataset images based on the provided .csv files
├── utils/                              
│   └── logging_config.py               # logging configuration
├── main.py                             # Main
├── train.py                            # Training loop
├── wandb_logger.py                     # Weights & Biases logging utilities
```

--- 

### Running the Project 
1. Open `FER-k-fold-CV.ipynb` in Google Colab and run all the cells. 

--- 

## License 
This project was made for educational purposes.