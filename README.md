# Persistent Activity and Anomaly Detection

A Human Activity Recognition (HAR) and motion-anomaly detection project built around smartphone inertial-sensor data from the **UCI Human. Activity Recognition Using Smartphones (UCI HAR) Dataset**.
The project reproduces and evaluates four activity-classification baselines, then uses a CNN feature extractor and an unsupervised autoencoder to identify unusual motion patterns. It also includes context-aware anomaly thresholds, temporal filtering, sensor-level gradient attribution, and a Streamlit-based interactive demonstration generated from the notebook.

## Table of Contents
-  [Project Overview]
-  [Key Features]
-  [Dataset]
-  [System Workflow]
-  [Models]
-  [Reported Results]
-  [Repository Contents]
-  [Requirements]
-  [Run the Project]
-  [Outputs]
-  [Important Methodological Notes]
-  [References]

## Project Overview
Human Activity Recognition uses motion-sensor measurements to identify what a person is doing. This project works with accelerometer and gyroscope signals to classify six everyday activities and to detect motion windows whose learned representations differ unusually from 
normal activity patterns.

The workflow has two related parts:

1. **Supervised activity recognition:** compare a Random Forest classifier with LSTM, GRU, and 1D-CNN models.

2. **Motion anomaly detection:** extract a compact representation with a 1D-CNN, reconstruct that representation with an autoencoder, and flag unusually large reconstruction errors using activity-specific thresholds and a temporal persistence filter.

The experiments are implemented in a Google Colab notebook. The notebook mounts Google Drive so that checkpoints, metrics, logs, and figures can persist beyond a temporary Colab runtime.

## Key Features
-  Loads both the 9-channel raw inertial signals and the 561 engineered features supplied by UCI HAR.
-  Preserves the official subject-independent training and test partitions.
-  Compares Random Forest, bidirectional LSTM, bidirectional GRU, and 1D-CNN classifiers.
-  Trains an autoencoder on CNN-derived latent representations.
-  Calculates reconstruction-error thresholds separately for each predicted activity context.
-  Applies a three-step hysteresis filter to reduce alerts caused by isolated error spikes.
-  Uses gradient-based saliency to estimate which sensor channels contributed most to a detected anomaly.
-  Generates evaluation metrics, confusion matrices, training curves, and a Streamlit demonstration.
-  Saves model checkpoints and experiment outputs to Google Drive.

## Dataset
The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.
The dataset contains recordings from 30 volunteers performing six activities while carrying a waist-mounted smartphone. Its inertial measurements include accelerometer and gyroscope signals.

### Activity classes
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

### Input representations
  --------------------- ---------------------------- ---------------------
  Representation                   Shape                  Used for
  --------------------- ---------------------------- ---------------------
  Raw inertial           9 channels × 128 time steps   LSTM, GRU, 1D-CNN and
  sequences                        per sample          anomaly pipeline

  Engineered feature     561 features per sample       Random Forest
  vectors                                            
 ------------------------------------------------------------------------

The notebook verifies the standard split sizes:
  ------------------------------
  Split        Number of samples
  ---------- -------------------
  Training                 7,352
  Testing                  2,947
  Total                   10,299
  ------------------------------
The six classes are encoded as classification targets. The official subject-independent split is retained rather than randomly mixing windows from the same subjects across training and testing.
The notebook clones the upstream implementation and obtains the dataset through that repository's data-preparation utilities:
- Upstream implementation: https://github.com/bravo2024/HAR

## System Workflow

### 1. Data preparation
-  Mount Google Drive.
-  Clone the upstream HAR repository.
-  Download or extract the UCI HAR dataset.
-  Load raw 9-channel sequences and 561-dimensional engineered features.
-  Verify dataset shapes and activity labels.

### 2. Baseline activity classification
Train and evaluate four classifiers:
-  Random Forest on engineered features.
-  Two-layer bidirectional LSTM on raw sequences.
-  Two-layer bidirectional GRU on raw sequences.
-  Three-block 1D-CNN on raw sequences.
Accuracy, macro-F1, per-class F1, and confusion matrices are calculated.

### 3. Latent feature extraction
The 1D-CNN is used as a feature extractor. Its compact latent representation provides the input to the anomaly-detection stage.

### 4. Autoencoder training
An unsupervised autoencoder learns to reconstruct the CNN latent representation. It uses mean squared reconstruction error as its training objective. The notebook trains for up to 50 epochs and uses early stopping based on training loss.

### 5. Context-aware anomaly scoring
For each test window, the pipeline obtains a CNN representation and computes its reconstruction error. An activity-specific threshold is selected using the training reconstruction-error distribution for that activity.
The notebook uses different standard-deviation multipliers for different activities:

  -------------------- ----------------
  Activity             Threshold rule
  -------------------- ----------------
  Walking              μ + 2.0σ
  Walking Upstairs     μ + 2.0σ
  Walking Downstairs   μ + 3.0σ
  Sitting              μ + 1.5σ
  Standing             μ + 3.0σ
  Laying               μ + 1.5σ
  -------------------- ----------------
Here, μ and σ denote the mean and standard deviation of the training reconstruction errors for the corresponding activity context.

### 6. Temporal hysteresis filtering
A single threshold breach may be caused by a short-lived sensor disturbance. The project therefore applies a hysteresis filter with a window size of three to distinguish isolated spikes from alerts that persist over time.
In the notebook's recorded test run:
-  Test windows evaluated: **2,947**
-  Raw threshold breaches: **513**
-  Alerts after temporal filtering: **117**
The notebook reports an approximately **77.2% reduction in alert count**after filtering. This is a reduction in threshold-breach events, not a measured real-world false-positive rate; ground-truth anomaly labels are not provided by the standard HAR activity labels.

### 7. Anomaly explanation
For a detected anomaly, gradient-based saliency is calculated with respect to the input sensor sequence. The absolute gradients are aggregated over time to estimate the contribution of each of the nine sensor channels to the reconstruction error.

### 8. Interactive demonstration
The notebook writes a Streamlit application and launches it from the Colab environment. The interface allows a user to inspect test windows, view the predicted activity and anomaly score, compare the score with the relevant context threshold, and inspect sensor-channel attribution when an anomaly is detected.
The Streamlit application is generated by notebook cells; it is not present as a separate tracked `app.py` file in the repository's top-level file listing.

## Models
  -----------------------------------------------------------------------
  Model                   Input                   Architecture /
                                                  configuration
  ----------------------- ----------------------- -----------------------
  Random Forest           561 engineered features 200 trees, maximum
                                                  depth 25

  LSTM                    9 × 128 raw sequence    Two-layer bidirectional
                                                  LSTM, hidden size 128,
                                                  dropout 0.3

  GRU                     9 × 128 raw sequence    Two-layer bidirectional
                                                  GRU, hidden size 128,
                                                  dropout 0.3

  1D-CNN                  9 × 128 raw sequence    Three convolutional
                                                  blocks (64, 128, 256
                                                  channels), global
                                                  average pooling

  Anomaly autoencoder     CNN latent vector       Fully connected
                                                  encoder-decoder;
                                                  256-dimensional input
                                                  and 64-dimensional
                                                  bottleneck
  -----------------------------------------------------------------------

The deep classification models use cross-entropy loss, AdamW optimization, and a cosine-annealing learning-rate scheduler. The notebook uses a fixed seed of 42 for reproducibility and trains the deep baselines for 15 epochs.
The autoencoder uses mean squared error, Adam optimization, a ReduceLROnPlateau scheduler, and early stopping.

## Reported Results
The following values are taken from the notebook's recorded baseline-reproduction output. Accuracy and macro-F1 are shown as proportions.
  --------------- --------------------- ---------------------
  Model             Reproduced accuracy   Reproduced macro-F1
  --------------- --------------------- ---------------------
  Random Forest                  0.9257                0.9238
  LSTM                           0.9128                0.9141
  GRU                            0.9141                0.9156
  1D-CNN                     **0.9410**            **0.9413**
  --------------- --------------------- ---------------------
The 1D-CNN achieved the strongest reproduced classification result among the four models in the recorded run.
These are notebook-recorded experimental results, not a guarantee that every run or hardware environment will produce identical values. Training time also depends on the runtime hardware.

## Repository Contents
The GitHub repository currently contains the following top-level artifacts:
  ------------------------------------------------------------------------------------------------
   File                                                        Description
  ------------------------------------------------------------ -----------------------------------
  `Persistent_HAR_Baseline (1).ipynb`                          Main Google Colab/Jupyter notebook
                                                               containing data preparation,
                                                               baseline reproduction, anomaly
                                                               detection, analysis, and Streamlit
                                                               app generation 
  
  `Persistent Activity and Anomaly Detection Report (2).pdf`   Project report

  `HAR_Anomaly_Detection_Presentation.pptx`                    Project presentation

  `image_no_yellow_outline.png`                                Image asset included in the
                                                               repository
  ------------------------------------------------------------------------------------------------
The notebook clones `bravo2024/HAR` at runtime to access the upstream HAR implementation and its dataset utilities. That upstream repository is a dependency of the notebook workflow; it is not a subdirectory of this repository.

## Requirements
The intended execution environment is **Google Colab**.
Main Python libraries used by the notebook include:
-  Python 3
-  PyTorch
-  NumPy
-  Pandas
-  scikit-learn
-  Matplotlib
-  Streamlit
-  Google Colab Drive integration
A CUDA-enabled GPU is recommended for faster deep-learning training, but the notebook includes CPU/GPU device detection and can fall back to CPU. Exact package versions may differ from the runtime used for the recorded experiment.

## Run the Project

### Option 1: Google Colab (recommended)
1. Open the repository: https://github.com/Surbhi1976/ML-project
2. Open `Persistent_HAR_Baseline (1).ipynb`.
3. Select **Open in Colab** if available, or download the notebook and upload it to Google Colab.
4. In Colab, select **Runtime → Change runtime type** and choose a GPU runtime if one is available.
5. Run the notebook cells in order, starting from the Google Drive mounting and environment setup cells.
6. Authorize Google Drive access when prompted.
7. Allow the notebook to clone the upstream HAR repository and prepare the dataset.
8. Continue through baseline training, autoencoder training, threshold generation, temporal filtering, attribution, and the Streamlit demonstration cells.

The notebook creates a Google Drive workspace at:
MyDrive/
└── Persistent Activity and Anomaly Detection/
    ├── out/
    ├── logs/
    └── figures/

### Option 2: Download the notebook
Clone this project and open the notebook locally in Jupyter:
git clone https://github.com/Surbhi1976/ML-project.git
cd ML-project
Open `Persistent_HAR_Baseline (1).ipynb` using Jupyter Notebook or JupyterLab.
**Note:** The notebook is designed around Google Colab paths, Drive mounting, and Colab-specific commands. A local run will require adapting those cells, providing persistent output directories, and ensuring the upstream HAR repository and dataset are available.

## Outputs:
The notebook saves or generates the following artifacts during execution:
-  `randomforest` evaluation metrics.
-  `lstm_best.pt`
-  `gru_best.pt`
-  `cnn1d_best.pt`
-  `autoencoder_best.pt`
-  `metrics.json`
-  Training-curve figures.
-  Confusion-matrix figures.
-  Context-aware anomaly thresholds and anomaly scores.
-  Sensor-channel gradient-attribution plots.
-  A generated Streamlit application bundle.
Most experiment artifacts are written to the Google Drive workspace described above. The exact set of generated files depends on which notebook cells are executed successfully.

## Important Methodological Notes
- **Checkpoint selection:** The reproduced baseline notebook follows the upstream implementation's practice of selecting deep-model checkpoints using test accuracy.  This exposes the test set to model-selection decisions and can make reported test performance optimistic. For a rigorous final evaluation, use a separate validation set for checkpoint selection and reserve the test set for one final evaluation.
- **Anomaly labels:** The standard UCI HAR labels describe activities, not verified anomalous events. The anomaly pipeline therefore detects deviations from learned activity representations; it should not be interpreted as a clinically validated or safety-critical anomaly detector.
- **Alert reduction:** The reported change from 513 raw breaches to 117 filtered alerts demonstrates temporal filtering of isolated events. It does not by itself establish precision, recall, or a 77.2% reduction in false positives.
- **Reproducibility:** Results may vary with hardware, library versions, random seeds, and runtime behavior. The notebook's recorded output reflects one execution.
- **Generated application:** The Streamlit app is assembled and launched by notebook code. Run the relevant notebook cells to generate it.

## References
1. **UCI Machine Learning Repository --- Human Activity Recognition Using Smartphones:**
    https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. **Upstream HAR implementation (`bravo2024/HAR`):**
    https://github.com/bravo2024/HAR
3. **This project repository:**
    https://github.com/Surbhi1976/ML-project
