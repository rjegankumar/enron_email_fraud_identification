# Identifying Persons of Interest in the Enron Scandal

![](proj_img.jpg)

One of the largest corporate scandals that resulted in the bankruptcy of a multi-billion dollar corporation was the [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal). The objective of this project is develop a machine learning model to identify Persons of Interest i.e. those who were indicted, agreed to a plea bargain etc. in this scandal using the data from emails and financials that were recorded before.

## Files

| File name | Description |
| :--- | :--- |
| [data_exploration.py](data_exploration.py) | Python script to explore Enron's emails and financials data dictionary |
| [enron_documentary_notes.txt](enron_documentary_notes.txt) | Text file containing notes taken while watching [Enron: The Smartest Guys in the Room](http://www.imdb.com/title/tt1016268/) |
| [environment.yml](environment.yml) | Project environment required to run this code |
| [feature_format.py](feature_format.py) | Python script used to convert the data dictionary to a list of numpy arrays to be used in sklearn models |
| [feature_selection.py](feature_selection.py) | Python script used to create and visualize new features, and identify the strongest features |
| [final_project_dataset.pkl](final_project_dataset.pkl) | Enron data dictionary pickle file |
| [my_classifier.pkl](my_classifier.pkl) | Final classifier pickle file for testing |
| [my_dataset.pkl](my_dataset.pkl) | Final dataset pickle file for testing |
| [my_feature_list.pkl](my_feature_list.pkl) | Final feature list pickle file for testing |
| [nb_classifier.py](nb_classifier.py) | Python script with the Guassian Naive Bayes classifier |
| [poi_id.py](poi_id.py) | Python script with final classifiers for comparison |
| [project_summary.md](project_summary.md) | Markdown summarizing the entire project |
| [random_forest_clf.py](random_forest_clf.py) | Python script with the Random Forest classifier |
| [references.txt](references.txt) | Text file containing links to references used in this project |
| [tester.py](tester.py) | Python script with the code for final testing |

## Setup

- You must have [Anaconda](https://www.continuum.io/downloads) installed to run this code.
- Create a conda environment using [environment.yml](environment.yml) YAML file. More help on this can be found [here](https://conda.io/docs/using/envs.html#use-environment-from-file).

## License

The contents of this repository are covered under the [MIT License](LICENSE).
