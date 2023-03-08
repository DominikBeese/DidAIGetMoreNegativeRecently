[![](https://img.shields.io/badge/Python-3.10.6-informational)](https://www.python.org/)
[![](https://img.shields.io/github/license/DominikBeese/DidAIGetMoreNegativeRecently?label=License)](/LICENSE)
# Did AI get more negative recently?
Data and code for the paper ["Did AI get more negative recently?"](https://royalsocietypublishing.org/doi/abs/10.1098/rsos.221159) by Dominik Beese, Begüm Altunbaş, Görkem Güzeler, and Steffen Eger, RSOS 2023.

<img src="https://user-images.githubusercontent.com/111588769/223692479-3b13460e-a13c-4886-8f38-f5795b10b624.jpg" alt="average stance per year and domain" width="410px">


## Content
The repository contains the following elements:
 * 📂 [Data](/Data)
   * 📂 [Datasets](/Data/Datasets) of paper titles and abstracts
   * 📂 [Human Annotated Data](/Data/Human%20Annotated%20Data) regarding stance of a paper
   * 📂 [Model Predicted Data](/Data/Model%20Predicted%20Data) with predictions of our model for all [Datasets](/Data/Datasets)
 * 📂 [Model](/Model) used for the analysis
 * 📂 [Code](/Code) to train and apply models
 * 📂 [Analysis](/Analysis) code to generate the plots


## Citation
```
@article{DidAIGetMoreNegativeRecently,
          title = "Did {AI} get more negative recently?",
         author = "Dominik Beese and Beg{\"u}m Altunba{\c{s}} and G{\"o}rkem G{\"u}zeler and Steffen Eger",
        journal = "Royal Society Open Science",
         volume = "10",
         number = "3",
          pages = "221159",
           year = "2023",
            doi = "10.1098/rsos.221159",
            URL = "https://royalsocietypublishing.org/doi/abs/10.1098/rsos.221159",
         eprint = "https://royalsocietypublishing.org/doi/pdf/10.1098/rsos.221159",
      publisher = "The Royal Society Publishing",
}
```
> **Abstract:** In this paper, we classify scientific articles in the domain of natural language processing (NLP) and machine learning (ML), as core subfields of artificial intelligence (AI), into whether (i) they extend the current state-of-the-art by the introduction of novel techniques which beat existing models or whether (ii) they mainly criticize the existing state-of-the-art, i.e. that it is deficient with respect to some property (e.g. wrong evaluation, wrong datasets, misleading task specification). We refer to contributions under (i) as having a ‘positive stance’ and contributions under (ii) as having a ‘negative stance’ (to related work). We annotate over 1.5 k papers from NLP and ML to train a SciBERT-based model to automatically predict the stance of a paper based on its title and abstract. We then analyse large-scale trends on over 41 k papers from the last approximately 35 years in NLP and ML, finding that papers have become substantially more positive over time, but negative papers also got more negative and we observe considerably more negative papers in recent years. Negative papers are also more influential in terms of citations they receive.
