# LOGComments_ItML_finalproject

Base is oop project at an early stage, most of it is useless (see the whole app subfolder) but I did not want to bother with removing it and fixing the errors, that can be done later.

```console
python logreg.py
```

will run the Logistic regression interface I wrote that allows the basic functioning needed for our purposes.

The folder **datasets** should be populated with any datasets we want to use, as the program will ask for its contents, and also the final results will be stored there.

TODO:
- [ ] Validation / Cross-Validation
- [ ] detect overfitting / underfitting
- [x] Metrics
- [x] Codebase cleanup
- [x] one-hot encoding not by sklearn
- [ ] regularisation to combat overfitting

Okay, it seems that we are not using cv for adjustments, and k-fold.
Problem -> the learning curves are fucked up.
We should implement regularisation.
We no longer rely on artifacts saved, we should fix it.
We have to cleanup the whale of logreg, split in etc.
Parameter tuning etc.

Observations:
-> Overfitting cannot be detected by analysis of non-loss values.
-> regularisation hyperparameters have to be tuned during VC (e.g. k-fold)
-> We compare validation/test loss and training loss

Can we use LogLoss?




Results from the first run:
Accuracy: 0.77
Precision: 0.75
Recall: 0.78
F1 Score: 0.76
ROC-AUC: 0.77
Log Loss: 7.84
Confusion Matrix:
[[276  82]
 [ 70 242]]

Results after deuselessification:
Accuracy: 0.77
Precision: 0.75
Recall: 0.77
F1 Score: 0.76
ROC-AUC: 0.77
Log Loss: 7.89
Confusion Matrix:
[[276  82]
 [ 71 241]]


---------
The percentage of all correct predictions (TP + TN) out of the total predictions
---------
Precision -> Out of all predicted positives, x% were actually positive.
---------
Recall -> Out of all actual positives, x% were correctly identified.
---------
F1 Score -> The harmonic mean of precision and recall.
---------
ROC-AUC -> Measures the ability of your model to separate classes. A score of 0.5 means random guessing, and closer to 1 is better.
---------
Log Loss -> The penalty for how far off your predicted probabilities are from the true labels. 0.1 - 1 is normal
---------
Confusion Matrix:
    confusion_matrix
    [TP   FP]"
    [FP   FN]
---------




