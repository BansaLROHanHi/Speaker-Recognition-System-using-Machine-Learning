import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.io import wavfile
from python_speech_features import mfcc, delta
from sklearn import preprocessing
import os                                       #imported all the required libraries and directories

def ftr_mat(path):                              #a function to extract feature matrix from the audio file
    smpl_rate,audio=wavfile.read(path)          #reading the audio file, audio and rate of sampling
    
    if len(audio.shape)==2:                     #If stereo, convert to mono by taking mean of channels
        audio=np.mean(audio, axis=1)
    
    mfcc_ftr=mfcc(audio,smpl_rate,             #the features for mfcc are hyperparameters, taken from AI 
                         winlen=0.025,         # 25ms window
                         winstep=0.01,         # 10ms step
                         numcep=20,            # number of cepstrum to return
                         nfilt=40,             # number of filters in mel filterbank
                         nfft=1200,            # FFT size
                         appendEnergy=True)    # append energy to mfcc vectors
    mfcc_norm=preprocessing.scale(mfcc_ftr)    #normalizing the mfcc features
    
    del_ftr=delta(mfcc_norm, 2)                #delta features are more important for speaker recognition rather than filter bank
    
    ftr=np.hstack([mfcc_norm, del_ftr])        #concatenating both mfcc and delta features
    return ftr

def folder(direc):                             #extracting features for a folder containing no. of speaker directories
    Ftrs=[]                                     
    for file in os.listdir(direc):             #speaker files in the directory
            path=os.path.join(direc,file)      #path of that audio file
            ftr=ftr_mat(path)                  #extracting features of the file by ftr_mat() function
            Ftrs.append(ftr)
    return np.vstack(Ftrs)                     #concatenate and return the features of all audio files of one speaker

def gmm(spkr_ftrs, K=16):                      #training GMM model on speaker features, K is here a hyperparameter
    model={}
    for spkr, ftr in spkr_ftrs.items():
        gmm=GaussianMixture(n_components=K, covariance_type='diag', random_state=42)   #covariance matrix is initialized diagonal
        gmm.fit(ftr)                           #feature fitting
        model[spkr]=gmm                        #model contains gmm for all speaker features
    return model

def identify(test, model):                     #function to identify speaker of a given testcase
    ftr=ftr_mat(test)                          #Extract features from test audio

    scores={}
    for speaker, gmm_model in model.items():
        scores[speaker]=np.sum(gmm_model.score_samples(ftr))      #score_samples() to calculate log-likelihood for each speaker's GMM

    res=max(scores, key=scores.get)           #score means log likelihhod, speaker having max likelihood is returned
    return res

print("Extracting features...\n")
spkr_ftrs={}
for direc in os.listdir("Speaker Data-Training"):   #opening directories of training dataset
    Path=os.path.join("Speaker Data-Training", direc)
    if os.path.isdir(Path):
        spkr_ftrs[direc]=folder(Path)               #by folder function feature matrix of each speaker is stored in spkr_ftrs
        print(f"Features shape of speaker: ",direc," : ", spkr_ftrs[direc].shape)

print("\nTraining GMM model.....\n")
model=gmm(spkr_ftrs)                         #training GMM for each speaker 
print("GMM models trained for all speakers.\n")

print("Testing the audios of different speakers: \n")
print("Audio File\t\tIdentified Speaker")
print("-" *40)

total=0
correct=0

for folder in os.listdir("Testing Data"):    #opening testing dataset
    path=os.path.join("Testing Data", folder)
    if os.path.isdir(path):
        for file in os.listdir(path):
            total += 1                      #for every audio file, total increments by 1
            audio=os.path.join(path, file)
            guess= identify(audio, model)   #guess is the identified speaker by the function
            
            if guess==folder:               #test audio file is stored in folder with the name of actual speaker
                correct+=1                  #if guessed speaker matches folder name, correct increments by 1
            
            print(f"{file}\t\t{guess}")

acc=(correct / total)*100              #accuracy calculated
print("\nAccuracy : ",acc,"%.")
print("Total test cases: ",total)
print("Correctly identified ",correct," test cases.")

print("\nUnidentified Test Cases\n")       #unidentified test set
print("Audio File\t\tIdentified Speaker")
print("-" *40)

for file in os.listdir("Unidentified TestCases"): #opening directory
    audio=os.path.join("Unidentified TestCases", file)
    guess=identify(audio, model)          #identified speaker
    print(f"{file}\t\t{guess}")

print("\nTesting complete.")                 
