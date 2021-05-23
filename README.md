# Safety Doors

Program for detecting people jammed by doors in trains.

# Installation instruction

# Install

## Setup python environment

* Install python3.7

```bash
sudo apt-get update

sudo apt-get install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install python3-pip (not sure if this nessesary)
sudo apt-get install python3.6
sudo apt-get install python3.6-venv
```

* Create virtual enviroment:
```bash
python3.6 -m venv venv
```
* Activate environment
```bash
source venv/bin/activate
```
* Install requirements:
```bash
venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt
```

# Usage

Analyze scene from .pcd file and generate output json
```bash
./scene_analyzer.py -i RES/example_clouds/10.pcd -o out.json
```

View scene with detected objects
```bash
./json_mapped.py -i RES/example_clouds/10.pcd -p out.json
```

# VoiceAlert

Tested on Windows 10 64x

* Install requirements:
* SpeechRecognition
```bash
   pip install SpeachRecognition 
```
* PyAudio for Python 3.7 and Windows 64x
```bash
   pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
```
* Scikit-learn 0.24.2
```bash
   pip install -U scikit-learn
```
# Usage
Set audio source.
We can get all possible sound sources from the list
```bash
   import speech_recognition as sr
   ...
    with sr.Microphone(device_index=1) as source:
    ...
```
device_index specified the device from which the sound will be read
