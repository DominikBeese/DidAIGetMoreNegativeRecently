# Human Annotated Data
This folder contains our human annotated NLP, ML and Hist datasets.
 * 🗃 [All human annotated papers](all.json)
 * 🗃 [`NLP` dataset](NLP.json)
 * 🗃 [`ML` dataset](ML.json)
 * 🗃 [`Hist` dataset](Hist.json)

Each json file contains a list of papers with the following keys:

| Key        | Value  | Description                                                          |
|------------|--------|----------------------------------------------------------------------|
| `url`      | string | url of the paper                                                     |
| `title`    | string | title of the paper                                                   |
| `year`     | number | year the paper was published                                         |
| `venue`    | string | venue the paper was published in                                     |
| `abstract` | string | abstract of the paper                                                |
| `stance`   | number | human annotated stance value between -1 (negative) and +1 (positive) |
