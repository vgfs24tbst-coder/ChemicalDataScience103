### 6. Interpretation & Discussion

- Interpret the EDA. Which patterns did you find, which anomalies, which correlations? What does the PCA tell you? Postulate some potential research questions and discuss potential hypotheses - use the results from your EDA as arguments.
- Report your findings in a separate text file (e.g. an `.md` file), which also needs to be submitted together with your script.


the pairplot indicates a correlation between proline content and malic acid. The magnesium content of class 1 contains 4 outliers. The confusion matrix shows that there is not a strong correlation between any two featues, save for flavanoids and total_phenols. 

The PCA tells us that there are two PCs that explain 36.2 repsectively 19.2% of the variance between the three target classes. In the scatter plot there is some overlap between the classes but they seem to form clusters. 

The logistic regression model had the best performance over all. Followed by rf and then dt. Log reg also performed the best on the 2D PCA featues. Meaning that the features selected were good. The ther models struggeled with them and had lower accuracy scores. Rf had a score of 1 on the test set -> over fittingg, model to complex. 

potential research questions: correlation between malic acid content and proline
why does the magnesium vary so much in wine class 1?

