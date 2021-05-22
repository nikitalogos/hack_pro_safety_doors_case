# Safety Doors

Program for detecting people jammed by doors in trains.

# Installation instruction

#Install

##Setup python environment

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




