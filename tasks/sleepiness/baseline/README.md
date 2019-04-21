# Time Series ML

## Setup Instructions

 - Add ```ComParE2019_ContinuousSleepiness.zip``` to this directory ([Download Link](https://megastore.uni-augsburg.de/get/InXJXZESS8/))

 - ```unzip ComParE2019_ContinuousSleepiness.zip```

 - Rename the unzipped directory to ```data```

## Running Baselines

 - Activate virtual environment if using one, e.g. ```source ~/ml_env/bin/activate```

 - To run ComParE baseline: ```cd data```, then ```python3 baseline.py```

 - To run our baseline: ```python3 main.py -data ComParE -model MLP```

## Running Other Models

- To run CNN: ```python3 main.py -data wav -model SoundClassifier```