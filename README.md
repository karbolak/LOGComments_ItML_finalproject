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




