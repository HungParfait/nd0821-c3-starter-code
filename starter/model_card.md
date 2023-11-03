# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistic Regression were trained.

* Model version: 1.0.0
* Model date: 30 Oct 2023

## Intended Use
The model is capable of making predictions regarding income classes based on census data. These classes consist of two groups: those earning more than 50K and those earning 50K or less.

## Training Data
The UCI Census Income Data Set was used for training. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
80% of the 32561 rows were used (26561 instances) in the training set for training.

## Evaluation Data
20% of the 32561 rows were used (6513 instances) in the test set for evaluation.

## Metrics
Three metrics were used for model evaluation (performance on test set):
* precision: 0.7110332749562172
* recall: 0.2617666021921341
* fbeta: 0.3826578699340245

## Ethical Considerations
- Since the dataset consists of public available data with highly aggregated census data no harmful unintended use of the data has to be addressed.
- Based on correlation matrix between race feature and target label. So I EDA and see the feature importance of the ML model
## Caveats and Recommendations
It should be better if a larger dataset is used.
