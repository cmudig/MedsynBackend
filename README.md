# MedsynBackend
This Repository contains the backend Server for the generative AI extension of the OHIF viewer. See https://github.com/TomWartm/Viewers for help.

## Model Weights
For model weights see in google drive: ```Generative AI in Radiology > MedSyn Model Parameter > MedSyn``` or in ```https://github.com/batmanlab/MedSyn```

## Run 
1. Clone this repository (on a machine with large GPU RAM)
    - `git clone https://github.com/TomWartm/MedsynBackend`
2. Navigate to the cloned project's directory
3. Install required python packages `conda env create --file environment.yml`
4. Actiave environment `conda activate medsyn-3-8`
5. Navigate to src folder
6. Run flask server `python app.py`


## Network configurations
* This is deplloyed using nginx and flask. 
* To see the status of the nginx, `sudo systemctl status nginx` and to see the status of the flask app `sudo systemctl status flask_app`
* To modify the flask app configurations go to `/etc/systemd/system/flask_app.service`
* To run the gunicorn setup not through flask and nginx, you should first stop the fllask_app `sudo systemctl stop flask_app` then go to `~/MedsynBackend/src/`, activate the medsyn-3-8 conda environment if it's not activated then run this command `/home/exouser/miniconda3/envs/medsyn-3-8/bin/gunicorn --workers 1 --bind 0.0.0.0:5000 app:app`