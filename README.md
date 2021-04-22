# random-forest-predictor
A Python script that tries to predict whether or not a person is depressed using RandomForest algorithm. It's intended to work with b_depressed.csv dataset I found on kaggle, but it should work with pretty much any other set as well. Outputs a handful of .pngs visualising each and every decision tree in whatever directory the script's being called. Obviously uses sklearn.

**Use the following command line parameters:** 

```
'-t',
dataset .csv file directory 
required

'-s',
determines how many rows will be used for tests. Between 0.2 and 0.8. 0.2 by default

'-c',
name of a column in a dataset that contains labels / classes
required

'--max_depth',
determines maximum depth of a tree. 5 by default

'--acceptable_impurity',
the level of impurity at which nodes are no longer split. 0 by default
```

**Example decision tree output .png:**

![exp](https://user-images.githubusercontent.com/52709292/115715481-75a3bc80-a378-11eb-9619-384b05551363.png)
