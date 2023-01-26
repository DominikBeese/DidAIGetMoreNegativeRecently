# Model Predicted Data
This folder contains our NLP and ML datasets as well as the dataset of accepted/rejected papers with stance values predicted by our model.
 * 🗃 [`NLP` dataset](NLP-Predictions.json)
 * 🗃 [`ML` dataset](ML-Predictions.json)
 * 🗃 [Dataset of accepted/rejected papers](AcceptedRejected-Predictions.json)


## `NLP` and `ML` datasets

Each json file contains a list of papers with the following keys:

| Key         | Value  | Description                                                          |
|-------------|--------|----------------------------------------------------------------------|
| `url`       | string | url of the paper                                                     |
| `title`     | string | title of the paper                                                   |
| `author`    | string | author(s) of the paper                                               |
| `year`      | number | year the paper was published                                         |
| `venue`     | string | venue the paper was published in                                     |
| `citations` | number | (optional) number of citations the paper has received, if known      |
| `stance`    | number | model predicted stance value between -1 (negative) and +1 (positive) |


## Dataset of accepted/rejected papers

The dataset contains papers compiled by us and papers compiled by [Kang et al. (2018)](https://aclanthology.org/N18-1149/) ([allenai/PeerRead](https://github.com/allenai/PeerRead)).

Each json file contains a list of papers with the following keys:

| Key      | Value  | Description                                                          |
|----------|--------|----------------------------------------------------------------------|
| `url`    | string | (optional) url of the paper, if known                                |
| `title`  | string | title of the paper                                                   |
| `author` | string | author(s) of the paper                                               |
| `year`   | number | year the paper was published                                         |
| `venue`  | string | venue the paper was published in                                     |
| `status` | string | `"accepted"` if the paper was accepted, `"rejected"` otherwise       |
| `stance` | number | model predicted stance value between -1 (negative) and +1 (positive) |
