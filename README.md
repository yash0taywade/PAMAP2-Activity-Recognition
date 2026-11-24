# PAMAP2 Activity Recognition

A machine learning pipeline for recognizing human physical activities from wearable sensor data using the PAMAP2 Physical Activity Monitoring dataset.

The pipeline loads raw sensor data, audits and cleans the data, segments continuous signals into time windows, extracts statistical features, trains a classifier, and evaluates activity recognition using a subject-held-out test split.

---

## 1. Five-Minute Plan

| Role            | Responsibility                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Apoorva Trivedi | Download the dataset, load raw files, clean missing values, and build the windowing/feature pipeline.                   |
| Yash Taywade    | Train the classifier, evaluate using a subject-held-out split, and write the interpretation of metrics and limitations. |

This plan was defined before implementation and was not changed during the task.

---

## 2. Project Goal

The goal is to build a working machine learning pipeline that:

1. Loads the raw sensor data.
2. Identifies and handles data-quality issues.
3. Cuts continuous sensor signals into time windows.
4. Extracts features from each window.
5. Trains a machine learning model to predict physical activity labels.
6. Reports model performance.
7. Explains why the evaluation result is meaningful.

---

## 3. Dataset

### PAMAP2 Physical Activity Monitoring

The project uses the **PAMAP2 Physical Activity Monitoring dataset**.

The dataset contains data from:

* 9 subjects
* 3 inertial measurement units (IMUs)

  * Wrist
  * Chest
  * Ankle
* A heart-rate monitor
* Multiple physical activity labels
* Raw timestamped sensor measurements
* Missing values represented as `NaN`

The IMU sensors operate at approximately 100 Hz, while the heart-rate monitor samples at approximately 9 Hz.

Dataset source:

https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+mon

### Why PAMAP2?

PAMAP2 was selected because it is the primary dataset specified in the task brief and provides subject-level data suitable for evaluating whether the model can generalize to a subject that was not present during training.

---

## 4. Data Quality and Cleaning

Before cleaning, the raw dataset contained missing sensor values and rows without an activity label.

The pipeline performs a raw-data audit before cleaning so that these issues can be quantified.

### Observed raw-data issues

For the final run:

* Total rows: **1,942,872**
* Total sensor values: **149,371,716**
* Missing sensor values: **3,093,966**
* Missing-value fraction: **2.0713%**
* Rows with zero activity label: **929,661**

Missing sensor values are handled during preprocessing, and rows without a valid activity label are excluded from activity-window generation.

The purpose of the cleaning stage is to prevent missing sensor measurements and unlabeled observations from being treated as valid supervised-learning examples.

---

## 5. Windowing and Feature Extraction

The raw sensor data is a continuous time series. Instead of making a prediction from individual sensor rows, the signal is divided into fixed-length time windows.

### Final configuration

* Window size: **5 seconds**
* Step size: **2.5 seconds**
* Windows therefore overlap by 50%.

Each window contains a sequence of sensor measurements. Statistical features are then extracted from the sensor values within the window.

The resulting representation is:

```text
Raw Sensor Signal
        ↓
Data Cleaning
        ↓
5-second Time Windows
        ↓
Feature Extraction
        ↓
Window-level Feature Vector
        ↓
Machine Learning Model
        ↓
Activity Prediction
```

Windowing is performed over contiguous activity runs so that activity boundaries are not unnecessarily mixed within a generated window.

---

## 6. What this model cannot do

### Yash Taywade

* It cannot claim generalization beyond the subject-held-out test subject without additional evaluation on other unseen subjects.
* It cannot reliably recognize activities that are not represented in the training labels.
* It cannot recover exact raw-sequence timing once the signal has been reduced to window-level features.
* It cannot guarantee strong performance on very short or heavily mixed-label windows.

### Apoorva Trivedi

* It cannot explain sensor failures that are not visible in the raw data columns.
* It cannot reliably identify activity transitions when the selected window contains signals from multiple activities.
* It cannot replace a full per-subject evaluation with a single overall accuracy score.
* It cannot guarantee the same performance if the train/test split, window size, step size, or held-out subject is changed.

---

## 7. Train/Test Split

The project uses a **subject-held-out evaluation strategy**.

### Training subjects

* 101
* 102
* 103
* 104
* 105
* 106
* 107
* 109

### Test subject

* 108

Subject 108 is completely held out from training and is used only for testing.

### Why subject-held-out evaluation?

A random row-level split could place highly similar observations or windows from the same person into both the training and test sets. This can make the reported performance look better than the model's ability to generalize to a new person.

By holding out an entire subject, the model is evaluated on data from a person it did not see during training.

This makes the evaluation more representative of the intended generalization setting.

---

## 8. Model and Evaluation

The extracted window-level features are used to train a tabular machine learning classifier.

The final evaluation reports:

* Accuracy
* Balanced accuracy

### Why balanced accuracy?

The activity classes are not perfectly balanced. In addition to ordinary accuracy, balanced accuracy provides a class-balanced view of performance by accounting for performance across the different activity classes.

---

## 9. Run Results

The final quick run used:

```text
Dataset: PAMAP2
Number of subjects: 9
Window size: 5 seconds
Step size: 2.5 seconds
Test subject: 108
```

### Dataset audit

| Metric                |       Value |
| --------------------- | ----------: |
| Total rows            |   1,942,872 |
| Total sensor values   | 149,371,716 |
| Missing sensor values |   3,093,966 |
| Missing fraction      |      0.0207 |
| Zero-labeled rows     |     929,661 |

### Model evaluation

| Metric            |     Result |
| ----------------- | ---------: |
| Training windows  |      6,583 |
| Test windows      |      1,028 |
| Accuracy          | **66.15%** |
| Balanced accuracy | **69.41%** |

The reported test performance comes from the subject-held-out evaluation using subject **108** as the test subject.

---

## 10. Why We Trust the Accuracy

The reported accuracy is based on a subject-held-out evaluation rather than a random row-level split.

Subject 108 was completely excluded from the training data and was used only for testing. Therefore, the test set represents observations from a subject that the model did not see during training.

This reduces the risk of leakage from having highly correlated observations from the same subject appear in both training and test sets.

Balanced accuracy is also reported alongside ordinary accuracy because the activity labels are not perfectly balanced.

The result should still be interpreted as an evaluation on **one held-out subject**, not as a complete estimate of performance across every possible new subject.

--- 

## 11. Project Structure 

```text
pamap2-activity-recognition/
│
├── data/
│   └── PAMAP2_Dataset/
│
├── src/
│   ├── pamap2_activity_recognition/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── model.py
│   │   └── pipeline.py
│   └── train.py
│
├── outputs/
│   ├── summary.json
│   └── metrics.json
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 12. How to Run

### 1. Install dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place the dataset

Extract the PAMAP2 dataset under the `data` directory.

The dataset should contain subject files such as:

```text
subject101.dat
subject102.dat
subject103.dat
...
subject109.dat
```

### 3. Run the pipeline

```bash
python3 main.py --data-root data --output-dir outputs --window-seconds 5 --step-seconds 2.5
```

### 4. View the results

After execution, the generated output files can be inspected under:

```text
outputs/
```

The pipeline reports the dataset profile, missing-data statistics, number of generated windows, test subject, accuracy, and balanced accuracy.

---

## 13. Example Output

```text
{
    'dataset': 'pamap2',
    'num_subjects': 9,
    'num_windows_train': 6583,
    'num_windows_test': 1028,
    'window_seconds': 5.0,
    'step_seconds': 2.5,
    'test_subject': 108,
    'accuracy': 0.6614785992217899,
    'balanced_accuracy': 0.6940513503595934
}
```

This corresponds to approximately:

```text
Accuracy           = 66.15%
Balanced Accuracy  = 69.41%
```

---

## 14. Limitations and Future Improvements

Possible improvements include:

* Evaluating every subject using leave-one-subject-out validation.
* Comparing multiple machine learning classifiers.
* Performing hyperparameter tuning.
* Adding more time-domain and frequency-domain features.
* Investigating class-specific performance using a confusion matrix.
* Testing different window and step sizes.
* Evaluating robustness across different unseen subjects.

---

## 15. Conclusion

This project demonstrates an end-to-end wearable-sensor activity recognition pipeline.
The system converts raw PAMAP2 sensor signals into cleaned, window-level feature representations and uses these features to classify physical activities.
The final subject-held-out evaluation achieved:
**66.15% accuracy and 69.41% balanced accuracy on subject 108.**
The subject-held-out design provides a more meaningful evaluation of generalization to an unseen subject than a random row-level split.
