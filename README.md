# Did AI get more Negative Recently?
Data and code for the paper "Did AI get more Negative Recently?" by Dominik Beese, Begüm Altunbaş, Görkem Güzeler, and Steffen Eger.
> **Abstract:** In this paper, we classify scientific articles in the domain of natural language processing (NLP) and machine learning (ML), as core subfields of artificial intelligence (AI), into whether (i) they extend the current state-of-the-art by introduction of novel techniques which beat existing models or whether (ii) they mainly criticize the existing state-of-the-art, i.e., that it is deficient with respect to some property (e.g., wrong evaluation, wrong datasets, misleading task specification). We refer to contributions under (i) as having a "positive stance" and contributions under (ii) as having a "negative stance" (to related work). We annotate over 1.5k papers from NLP and ML to train a SciBERT based model to automatically predict the stance of a paper based on its title and abstract. We then analyze large-scale trends on over 41k papers from the last ~35 years in NLP and ML, finding that papers have gotten substantially more positive over time, but negative papers also got more negative and we observe considerably more negative papers in recent years. Negative papers are also more influential in terms of citations they receive.

## Content
The repository contains the following elements:
 * 📂 [Data](/Data)
   * 📂 [Datasets](/Data/Datasets) of paper titles and abstracts
   * 📂 [Human Annotated Data](/Data/Human%20Annotated%20Data) regarding stance of a paper
   * 📂 [Model Predicted Data](/Data/Model%20Predicted%20Data) with predictions of our model for all [Datasets](/Data/Datasets)
 * 📂 [Model](/Model) used for the analysis
 * 📂 [Code](/Code) to train and apply models
 * 📂 [Analysis](/Analysis) code to generate the plots
