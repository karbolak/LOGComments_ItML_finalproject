# TODO:
# - litania ifów, explained below
# - zmienić model attributes w deployment

# - go though instruction and checklist
# - documentation

# - zadecydować do idzie do raportu
# - ustalić co jest w grafie

# - visual representation
# - extra features
# - 11. OOP-011: As a datascientist, I would like to create a model run where
#   I can fine-tune an existing model on a new dataset.

# pipeline details: delete type, parameters if not needed, trained,
# display metrics in pipeline details (only eval ones)
# wymyślić jak przechowywać model metrics i je dodać do raportu

"""
What these classification models accept:
- logistic:
    - any values in parameter columns
    - only 0 and 1 in the results colums
    - if result column has 2 values, it can be accepted
    - if those values are not 0 and 1, add a button to determinw what is
      treated as 0 and what as 1
    - add a posibility to change function parameters
- knn:
    - I see no issues
    - just allow the user to choose the value of k
- treeclassifier:
    - just allow the user to choose max_depth

Regression models:
- multiple linreg:
    - we just assume params are independent
- treereg:
    - LGTM, just allow the user to choose max_depth
- ridge:
    - when a data set contains a higher number of predictor variables than the
      number of observations. The second-best scenario is when
      multicollinearity is experienced in a set
"""
