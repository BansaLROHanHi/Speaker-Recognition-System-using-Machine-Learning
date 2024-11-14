# Speaker Recognition System using Gaussian Mixture Model (GMM)

This project implements a speaker recognition system using Gaussian Mixture Models (GMM). The system is capable of identifying speakers based on their unique voice features, achieving 100% accuracy on test cases.

## Features
- **Accuracy**: 100% recognition accuracy in test cases.
- **Efficiency**: Processes test cases quickly with accurate speaker identification.
- **Adaptability**: Can be extended to add new speakers or modify existing parameters.

## Requirements
- Python
- Libraries:
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `python_speech_features`
  - `os` (part of Python standard library)

## Project Structure
- **Speaker Data-Training/**: Contains training data for each speaker, with each speaker’s audio files in separate folders named after the speaker.
- **Testing Data/**: Holds test audio files for each speaker, organized in subdirectories named after each speaker.
- **Unidentified TestCases/**: Contains audio files of unknown speakers to test the model’s ability to classify unseen speakers.
- **speaker_recognition.py**: The main script that handles feature extraction, model training, and testing.

## Code Explanation

### 1. Feature Extraction
- **Function**: `ftr_mat(path)`
- **Purpose**: Extracts MFCC and delta features from audio files, capturing each speaker’s unique vocal characteristics.
- **Process**:
  - Reads the audio file and converts it to mono if stereo.
  - Extracts MFCC features with specified parameters (e.g., window size, number of filters).
  - Computes delta features and normalizes MFCC values.
  - Concatenates MFCC and delta features into a single feature matrix.

```python
def ftr_mat(path):
    smpl_rate, audio = wavfile.read(path)
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    mfcc_ftr = mfcc(audio, smpl_rate, winlen=0.025, winstep=0.01, numcep=20, nfilt=40, nfft=1200, appendEnergy=True)
    mfcc_norm = preprocessing.scale(mfcc_ftr)
    del_ftr = delta(mfcc_norm, 2)
    ftr = np.hstack([mfcc_norm, del_ftr])
    return ftr
```

### 2. Training the GMM Model
- **Function**: `gmm(spkr_ftrs, K=16)`
- **Purpose**: Trains a Gaussian Mixture Model (GMM) for each speaker using their extracted feature set, creating a unique model that represents each speaker’s vocal patterns.
- **Process**:
  - Loops through each speaker's feature data in `spkr_ftrs`.
  - Initializes a GMM with `K` components (default is 16) and sets the covariance type to diagonal for computational efficiency.
  - Fits the GMM on the feature set of each speaker and stores each trained model in a dictionary.
  - Returns a dictionary where keys are speaker names, and values are their corresponding GMM models.

```python
def gmm(spkr_ftrs, K=16):
    model = {}
    for spkr, ftr in spkr_ftrs.items():
        gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=42)
        gmm.fit(ftr)
        model[spkr] = gmm
    return model
```
### 3. Speaker Identification
- **Function**: `identify(test, model)`
- **Purpose**: Identifies the speaker of a given test audio file by evaluating which speaker model (GMM) best matches the audio features.
- **Process**:
  - Extracts features from the test audio file using the `ftr_mat` function.
  - Initializes a `scores` dictionary to store the log-likelihood score for each speaker’s GMM.
  - Loops through each speaker’s GMM model in `model` and computes the log-likelihood of the test audio features for each model using `score_samples()`.
  - Selects the speaker with the highest score (highest likelihood) as the identified speaker.
  - Returns the identified speaker’s name as the result.

```python
def identify(test, model):
    ftr = ftr_mat(test)
    scores = {speaker: np.sum(gmm_model.score_samples(ftr)) for speaker, gmm_model in model.items()}
    return max(scores, key=scores.get)
```
### 4. Testing and Accuracy Calculation
- **Purpose**: Evaluates the model's performance on a test dataset by comparing predicted speaker labels to actual labels and calculating the accuracy of speaker identification.
- **Process**:
  - Initializes counters for `total` and `correct` test cases.
  - Iterates through each speaker folder in the `Testing Data` directory.
  - For each test audio file in a speaker’s folder:
    - Increments the `total` counter.
    - Calls the `identify` function to predict the speaker for the test audio.
    - Compares the predicted speaker to the actual speaker (folder name). If they match, increments the `correct` counter.
    - Prints each audio file’s name and the identified speaker.
  - After testing all files, calculates accuracy as `(correct / total) * 100`.
  - Prints the final accuracy, total test cases, and correctly identified cases.

```python
total, correct = 0, 0
for folder in os.listdir("Testing Data"):
    path = os.path.join("Testing Data", folder)
    if os.path.isdir(path):
        for file in os.listdir(path):
            total += 1
            audio = os.path.join(path, file)
            guess = identify(audio, model)
            if guess == folder:
                correct += 1
            print(f"{file}\t\t{guess}")

accuracy = (correct / total) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Total test cases: {total}")
print(f"Correctly identified: {correct}")
```
## Usage
- **Step 1**: Place each speaker's audio files in `Speaker Data-Training/`, with each speaker's audio in a separate folder named after the speaker (e.g., `Speaker Data-Training/Abhay/`, `Speaker Data-Training/Eknath/`).
- **Step 2**: Place test audio files in `Testing Data/`, organizing them in subdirectories named after each speaker (e.g., `Testing Data/Abhay/`).
- **Step 3**: For testing unknown speakers, place their audio files in `Unidentified TestCases/`.

To run the speaker recognition system:
1. **Execute the script**:
   ```bash
   python speaker_recognition.py```
   
## Sample Output
When you run the `speaker_recognition.py` script, the output will include details about feature extraction, GMM model training, and the results of speaker identification for each test audio file. Below is an example of the expected output.

### Example Output
Shown for only six speakers here...
```plaintext
Extracting features...

Features shape of speaker: Abhay : (2717, 40)
Features shape of speaker: Eknath : (3037, 40)
Features shape of speaker: Rg : (2547, 40)
Features shape of speaker: Rishika : (3037, 40)
Features shape of speaker: Shivam : (4481, 40)
Features shape of speaker: Vaibhav : (3162, 40)

Training GMM model.....

GMM models trained for all speakers.

Testing the audios of different speakers:

Audio File              Identified Speaker
----------------------------------------
Abhay_15.wav            Abhay
Abhay_16.wav            Abhay
Abhay_17.wav            Abhay
Eknath_audio1.wav       Eknath
Rg_16.wav               Rg
Rishika_audio1.wav      Rishika
chappu_10.wav           Shivam
Vaibhav_16.wav          Vaibhav
...

Accuracy : 100.0 %.
Total test cases: 51
Correctly identified: 51

Unidentified Test Cases

Audio File              Identified Speaker
----------------------------------------
P1.wav                  Abhay
P2.wav                  Eknath
P3.wav                  Rg
P4.wav                  Rishika
P5.wav                  Shivam
P6.wav                  Vaibhav

Testing complete.
```
## Parameters
### MFCC & Delta Features:
Extracted using python_speech_features for improved speaker recognition.

### GMM Components (K):
The number of GMM components (default is 16). Increase K for more complex models if necessary.
