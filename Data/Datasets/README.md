# Datasets
This folder contains our NLP and ML datasets as well as the dataset of accepted/rejected papers.
 * 🗃 [`NLP` dataset](NLP.json)
 * 🗃 [`ML` dataset](ML.json)
 * 🗃 [Dataset of accepted/rejected papers](AcceptedRejected.json)


## `NLP` and `ML` datasets

Each json file contains a list of papers with the following keys:

| Key         | Value  | Description                                                     |
|-------------|--------|-----------------------------------------------------------------|
| `url`       | string | url of the paper                                                |
| `title`     | string | title of the paper                                              |
| `author`    | string | author(s) of the paper                                          |
| `year`      | number | year the paper was published                                    |
| `venue`     | string | venue the paper was published in                                |
| `abstract`  | string | abstract of the paper                                           |
| `citations` | number | (optional) number of citations the paper has received, if known |


## Dataset of accepted/rejected papers

The dataset contains papers compiled by us and papers compiled by [Kang et al. (2018)](https://aclanthology.org/N18-1149/) ([allenai/PeerRead](https://github.com/allenai/PeerRead)).

Each json file contains a list of papers with the following keys:

| Key        | Value  | Description                                                    |
|------------|--------|----------------------------------------------------------------|
| `url`      | string | (optional) url of the paper, if known                          |
| `title`    | string | title of the paper                                             |
| `author`   | string | author(s) of the paper                                         |
| `year`     | number | year the paper was published                                   |
| `venue`    | string | venue the paper was published in                               |
| `abstract` | string | abstract of the paper                                          |
| `status`   | string | `"accepted"` if the paper was accepted, `"rejected"` otherwise |
