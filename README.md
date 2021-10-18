# Capstone_Project
Udacity Data Science Nanodegree capstone project - arvato dataset

As part of the Udacity Data Science Nanodegree, for the final capstone project I have chosen to work with data from a mail-order company. The data available is as follows:
  1. Demographics data for a sample of ~900k individuals from the general population of Germany
  2. Demographics data for ~190k customers of a mail-order company
  3. Demographics data for ~86k individuals subject to a marketing campaign.
Data is not available via github, only through the Udacity course.

The demographics data includes 366 features, varying from information at an individual level to grouped-level data about their household, building and neighbourhood, and includes mostly  categorical with some continuous variables. The marketing campaign data is equally split into a test and train set, where the labels of whether the marketing campaign was successful are withheld for the test set. In addition to the data, data dictionaries are also provided, giving insight into the meaning of features and their values.

The aims of the project are two-fold:

  - Product a customer-segmentation report, outlining how the customer population (data source 2) differs from the general population of Germany (data source 1). This must be done using unsupervised learning techniques.
  - Use supervised learning techniques  to predict which customers are likely to respond positively to a marketing campaign run by the mail-order company (data source 3).

Included in this repository is a notebook of the steps I have taken to acheive these goals, as well as a .py file containing functions used. A write-up of the project can be found at https://clare-j.medium.com/mail-order-company-customer-analysis-c4ebb8a6272e where the approach taken is walked through and the steps align with the notebook.
Libraries used:
  - numpy
  - pandas
  - matplotlib.pyplot
  - pickle
  - prince
  - sklearn
  - imblearn

Credit to Udacity for the project, data and support.
