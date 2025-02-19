# Amazon_consumer_analysis
This repository offers a comprehensive framework for analyzing Amazon consumer behavior by merging transaction data with survey responses. By combining econometric methods, machine learning, clustering, and causal inference, the analyses provide both predictive insights and causal interpretations.
Below is a GitHub-ready README file in Markdown format that summarizes and explains your analysis scripts in a detailed, structured manner. You can copy and paste the text into your repository's README.md file.

---

# Amazon Consumer Behavior Analysis

This repository contains comprehensive analyses of Amazon consumer behavior by merging purchase records with survey responses. The project explores multiple dimensions of the data using econometric panel methods, predictive modeling, clustering, event study, and causal inference techniques.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Data Preprocessing](#data-preprocessing)
- [Panel Data and Regression Analysis](#panel-data-and-regression-analysis)
- [Predictive Analytics and Machine Learning](#predictive-analytics-and-machine-learning)
- [Clustering and Segmentation](#clustering-and-segmentation)
- [Event Study and Difference-in-Differences](#event-study-and-difference-in-differences)
- [Causal Analysis with DoWhy](#causal-analysis-with-dowhy)
- [Visualization and Interpretation](#visualization-and-interpretation)
- [How to Run the Analyses](#how-to-run-the-analyses)
- [License](#license)

## Overview

The goal of this project is to understand the relationship between Amazon consumer purchase behavior and various factors including demographics, self-reported usage, and life events. The analysis covers:

- **Data Preprocessing & Aggregation:** Merging Amazon purchase data with survey responses and creating panel data.
- **Panel Regression Analysis:** Employing fixed effects and random effects models to explore spending and frequency.
- **Predictive Modeling:** Forecasting purchase frequency using machine learning (Random Forests, Gradient Boosting) with interpretation via SHAP.
- **Clustering:** Segmenting consumers based on aggregated purchasing behavior and survey responses using K-Means.
- **Event Study & Difference-in-Differences (DiD):** Assessing the impact of life events (e.g., “Lost a job”) on spending behavior.
- **Causal Inference:** Utilizing DoWhy to build causal models, identify causal effects, and test their robustness.

## Repository Structure

- **amazon_analysis1.py:**  
  Contains scripts for data loading, merging, panel data creation, regression analysis, machine learning predictive models, clustering, and visualization.  
  Key methods include:
  - Data preprocessing and panel data regression.
  - Machine learning model training and interpretation (SHAP).
  - Clustering and subgroup analysis.

- **analysis_amazon_part_2.py:**  
  Focuses on causal analysis using the DoWhy framework.  
  Key steps include:
  - Creating life change dummy variables.
  - Merging and aggregating customer-level data.
  - Conducting event study and DiD analyses.
  - Specifying, estimating, and refuting causal models.

- **Additional Files:**  
  Any output plots, images (e.g., causal graph PNG files), or supplementary documentation generated during analysis.

## Installation and Requirements

To run the analyses, ensure you have Python 3.7+ and install the required packages. You can install the dependencies using:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn linearmodels shap econml dowhy
```

If using Google Colab or Jupyter Notebook, update file paths to match your data locations (e.g., Google Drive).

## Data Preprocessing

### Merging Datasets

- **Amazon Purchases:**  
  Loaded from a CSV file with order details such as `Order Date`, `Purchase Price Per Unit`, `Quantity`, and product information.
  
- **Survey Data:**  
  Contains self-reported usage metrics, demographic information, and responses to life events (e.g., "Q-life-changes").

- The datasets are merged on the common key (e.g., `Survey ResponseID`), and additional features such as total spending (price × quantity) are computed.

### Feature Engineering

- Aggregation at the user level (or panel level) calculates key metrics:
  - **Frequency:** Count of orders per user per period.
  - **Spending:** Total spending per user per period.
- Dummy variables are created for categorical demographic variables and survey responses.

## Panel Data and Regression Analysis

- **Fixed Effects Models:**  
  Control for time-invariant heterogeneity using individual (entity) effects.
  
- **Random Effects Models:**  
  Provide an alternative estimation for comparison.
  
- Models are estimated for both purchase frequency and spending, with detailed summary outputs printed to the console.

## Predictive Analytics and Machine Learning

- **Random Forest and Gradient Boosting Regressors:**  
  Predict actual purchase frequency using a set of survey features.
  
- **Model Evaluation:**  
  Metrics such as Mean Squared Error (MSE), R², and cross-validation RMSE are computed.
  
- **Model Interpretation:**  
  SHAP is used to interpret the importance and impact of each feature on the predictions.

## Clustering and Segmentation

- **K-Means Clustering:**  
  Segments consumers based on aggregated purchase metrics (total spend, frequency, average price, category diversity) and survey responses.
  
- **Silhouette Analysis:**  
  Determines the optimal number of clusters.
  
- **Visualizations:**  
  Boxplots and scatterplots illustrate differences between clusters.

## Event Study and Difference-in-Differences

- **Event Study:**  
  Evaluates dynamic changes in spending around the occurrence of life events.
  
- **Difference-in-Differences (DiD):**  
  Estimates the causal impact of a life event by comparing pre- and post-event spending.
  
- Visualizations include error bar plots showing treatment effects over time.

## Causal Analysis with DoWhy

- **Life Change Dummy Variables:**  
  Process survey responses using `MultiLabelBinarizer` to create one-hot encoded variables for life events.
  
- **Causal Model Specification:**  
  A Directed Acyclic Graph (DAG) is constructed to model relationships among treatment (e.g., "Lost a job"), outcome (e.g., total purchase), and confounders (demographics).
  
- **Identification and Estimation:**  
  The back-door criterion is applied to identify the causal estimand, which is then estimated using linear regression methods.
  
- **Refutation Tests:**  
  The model’s robustness is tested with methods like random common cause and data subset refuters.

## Visualization and Interpretation

- **Descriptive Plots:**  
  Histograms, boxplots, scatterplots, and pairplots explore distributions and relationships in the data.
  
- **Correlation Heatmaps:**  
  Visualize correlations between spending and order metrics.
  
- **Causal Graphs:**  
  Generated with DoWhy to illustrate the assumed causal relationships.
  
- **ROC Curves:**  
  Evaluate classification models (e.g., logistic regression for "compulsive purchase").

## How to Run the Analyses

1. **Data Files:**  
   Update file paths in the scripts (e.g., for CSV files stored on Google Drive or local directories).

2. **Execution:**  
   Run the provided Python scripts or open the Jupyter/Colab notebooks sequentially.

3. **Output:**  
   Analysis outputs include regression summaries, SHAP plots, clustering visualizations, causal graphs, and model evaluation metrics.

4. **Customization:**  
   You can modify the feature lists, model parameters, or regression formulas to tailor the analysis to your research questions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Conclusion

This repository offers a comprehensive framework for analyzing Amazon consumer behavior by merging transaction data with survey responses. By combining econometric methods, machine learning, clustering, and causal inference, the analyses provide both predictive insights and causal interpretations. For questions or contributions, please open an issue or submit a pull request.

