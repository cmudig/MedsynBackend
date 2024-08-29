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
