# Lesson 6: Linear, Multiple Linear, and Logistic Regression

## Questions to Ponder

- What is the difference between simple linear, multiple linear, and logistic regression?
- How does brute-force optimization find the best model parameters?
- What is an objective function, and why do we minimize it?
- How do we detect overfitting using train-validation-test splits?
- What does $R^2$ actually tell us about a regression model? What does it *not* tell us?
- What is a confusion matrix, and why is accuracy alone insufficient for classification?
- What assumptions do these regression methods make, and what happens when those assumptions are violated?
- How does the curse of dimensionality relate to overfitting?

## Introduction

Imagine you are trying to predict tomorrow's high temperature. You have one piece of information: today's high temperature. You suspect a linear relationship—hot today, hot tomorrow. This is **simple linear regression**.

Now imagine you have more information: today's temperature, today's humidity, today's wind speed, and the current barometric pressure. You want to combine all of them to make a better prediction. This is **multiple linear regression**.

Now imagine a different problem. You are not predicting a temperature. You are trying to predict whether it will rain tomorrow—yes or no. You still have the same measurements, but now your output is a category, not a number. This is **logistic regression**.

All three methods share a common foundation: they learn parameters by minimizing an objective function on training data. The differences lie in what they predict and how they measure error.

## Simple Linear Regression

Simple linear regression models the relationship between a single independent variable $x$ and a dependent variable $y$ as:

$$y = mx + b + \epsilon$$

Here, $m$ is the slope, $b$ is the intercept, and $\epsilon$ (epsilon) represents noise—the random scatter that prevents any real dataset from falling perfectly on a straight line.

The goal is to find the $m$ and $b$ that minimize the **sum of squared errors (SSE)**:

$$SSE = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

where $y_i$ is the true value and $\hat{y}_i = m x_i + b$ is the predicted value. Why square the errors? Because squaring penalizes large errors more heavily than small ones. Being off by 10 contributes 100 to the SSE; being off by 1 contributes only 1.

### Brute-Force Optimization

The simplest way to find the best $m$ and $b$ is to try many possibilities and pick the best one.

Imagine searching over all slopes between 0 and 2 in steps of 0.1, and all intercepts between -1 and 1 in steps of 0.1. For each combination, you compute the SSE. The combination with the smallest SSE wins.

This approach is computationally expensive—imagine doing this with 100 features instead of one!—but it illustrates the fundamental principle: **model fitting is an optimization problem over parameter space.**

### Efficient Methods

In practice, you will not use brute-force search. NumPy provides `polyfit(x, y, deg=1)`, which computes the optimal coefficients using a closed-form analytic solution. scikit-learn provides `LinearRegression`, which implements the same solution with a convenient API. You instantiate the model, call `.fit(X, y)`, and then access `.coef_` and `.intercept_`.

The `.score()` method returns $R^2$, which we will discuss shortly.

## Multiple Linear Regression

Multiple linear regression extends simple linear regression to multiple predictors:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

Here, $\beta_0$ is the intercept, each $\beta_i$ is a coefficient for predictor $x_i$, and $p$ is the number of features. The parameters are learned by minimizing the same SSE objective function.

**NB:** The matrix algebra version of this equation, $y = X\beta + \epsilon$, is worth studying if you continue in data science. It reveals why multiple linear regression has an analytic solution: $\hat{\beta} = (X^T X)^{-1} X^T y$.

### Train-Validation-Test Splits

Here is a mistake that catches even experienced practitioners. You fit a model. You evaluate it on the *same* data you used for training. The performance looks excellent. You deploy the model, and it fails immediately.

What happened? Your model memorized the training data, including its noise and quirks. It did not learn the underlying pattern. This is **overfitting**.

To detect overfitting, you must split your data into three non-overlapping subsets before doing anything else: a **training set**, a **validation set**, and a **testing set**.

But how big should each set be? You will find many textbooks and online tutorials that prescribe specific ratios: 80/10/10, 70/15/15, 60/20/20. These rules of thumb are not wrong, but they are not the point.

Let us think about what each set needs.

The **training set** is where the model learns its parameters. You want this set to be as large as possible. A larger training set gives the model more examples to learn from, which usually leads to better generalization. Every data point you hold back for validation or testing is a data point the model cannot learn from.

The **validation set** is where you monitor performance during training. You use it to tune hyperparameters, compare different models, and detect overfitting. You want this set to be as large as possible too. A larger validation set gives you a more reliable estimate of how the model is performing. If your validation set is too small, random noise can mislead you. A dip in validation accuracy might just be bad luck, not overfitting. A spike might be good luck, not genuine improvement.

The **testing set** is where you prove that your model works. You use it exactly once, at the very end of your project, to report final performance. You want this set to be as large as possible as well. A larger testing set gives you a more trustworthy final evaluation. If your testing set is too small, you cannot be confident that your model will perform well on new data.

Do you see the problem?

You have competing interests. You want all three sets to be as large as possible, but your total amount of data is fixed. Every observation you put into the validation set is an observation you cannot put into the training set. Every observation you put into the testing set is an observation you cannot put into either the training set or the validation set.

There is no perfect ratio that works for every situation. A massive dataset with millions of rows can afford to hold out 20% for validation and 20% for testing while still leaving plenty for training. A tiny dataset with only 100 rows cannot afford to hold out anything. In that case, you might need techniques like cross-validation, which we will discuss in a later lesson.

**NB:** The most important thing is not the specific ratio you choose. The most important thing is that **each set is representative of the population your data comes from.**

Imagine you are building a model to predict housing prices. Your data contains houses from five different cities. If you put all the houses from City A into the training set and all the houses from City B into the testing set, your evaluation will be meaningless. The model will learn the patterns specific to City A, and you will test it on a completely different distribution of houses. Your testing performance will tell you nothing about how the model would perform on a new house from City A—or even from City C.

The splits must preserve the underlying structure of your data. If your data has categories, each split should contain a representative proportion of those categories. If your data comes from different time periods, each split should span the same range of time.

In practice, you achieve this by using `train_test_split` with appropriate arguments: `stratify` for categorical targets, `shuffle` to randomize order, and `random_state` to make your results reproducible.

So, how should you choose your split sizes? There is no single correct answer. You must consider your total sample size, the complexity of your problem, the noise in your data, and the cost of making a mistake. A good starting point is to hold out 10-20% for validation and 10-20% for testing. Then adjust based on your specific situation.

But never forget: **the representativeness of your splits matters more than the exact ratios.**

### The $R^2$ Coefficient of Determination

$R^2$ measures the proportion of variance in the dependent variable explained by the model:

$$R^2 = 1 - \frac{RSS}{TSS}$$

where $RSS = \sum (y_i - \hat{y}_i)^2$ is the residual sum of squares (the error your model makes) and $TSS = \sum (y_i - \bar{y})^2$ is the total sum of squares (the error you would make by always predicting the mean).

$R^2$ ranges from $-\infty$ to 1:

- $R^2 = 1$: Perfect prediction. Your model explains all the variance.
- $R^2 = 0$: Your model performs no better than simply predicting the mean.
- $R^2 < 0$: Your model performs *worse* than predicting the mean. This is a clear sign of severe overfitting or a misspecified model.

### Overfitting in Practice

Consider a multiple linear regression model trained on a dataset with many features. The training $R^2$ might be 0.62—not perfect, but respectable. You might think the model is reasonably effective.

Then you evaluate the same model on the validation set. The validation $R^2$ is -2.80.

What happened? The model learned patterns specific to the training set that do not generalize. It found spurious correlations that exist only in the training data. When confronted with new data, those spurious correlations produced wildly wrong predictions.

**Training performance alone is meaningless.** Without validation, you would have confidently deployed a model that is worse than useless—worse, even, than simply predicting the average value every time.

## Logistic Regression

Now consider a different problem. You want to predict whether a customer will buy a product (yes or no). You have information about the customer: their age, their income, how many times they have visited your website.

Linear regression cannot help you here. It produces continuous outputs, not probabilities or categories. What you need is a method that outputs a probability between 0 and 1.

Logistic regression adapts the linear framework to binary classification. It predicts the probability that an observation belongs to the positive class:

$$P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p)}}$$

This is called the **sigmoid function** (or logistic function). It maps any real-valued input to a value between 0 and 1. Large positive inputs produce probabilities near 1. Large negative inputs produce probabilities near 0. An input of 0 produces a probability of 0.5.

To make a classification, you choose a threshold—typically 0.5. If $P(y=1) \geq 0.5$, predict "yes." If $P(y=1) < 0.5$, predict "no."

### Objective Function for Classification

Linear regression uses SSE as its objective. Logistic regression uses **cross-entropy loss** (also called log loss):

$$L = -\sum_{i=1}^{N} [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$

Let me translate. For each data point:

- If the true class is 1 ($y_i = 1$), the loss is $-\log(\hat{p}_i)$. If your predicted probability is close to 1, $\log(1) = 0$, so the loss is small. If your predicted probability is close to 0, $\log(0) \to -\infty$, so the loss is huge.
- If the true class is 0 ($y_i = 0$), the loss is $-\log(1 - \hat{p}_i)$. If your predicted probability is close to 0, the loss is small. If it is close to 1, the loss is huge.

Cross-entropy penalizes confident wrong predictions much more heavily than near-threshold errors. Unlike SSE, cross-entropy has no closed-form solution. It requires numerical optimization.

### Data Preprocessing

Before fitting logistic regression, you must prepare your data.

**One-hot encoding** converts categorical variables into binary indicator columns. Consider a "marital status" variable with three categories: married, divorced, and single. One-hot encoding creates three new columns: `marital_married` (1 if married, else 0), `marital_divorced` (1 if divorced, else 0), and `marital_single` (1 if single, else 0). This prevents the algorithm from assuming that "divorced" is halfway between "married" and "single."

**Min-max normalization** scales all numeric features to the range $[0, 1]$:

$$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$

Why is this necessary? Consider two features: income (ranging from \$0 to \$200,000) and age (ranging from 0 to 100). Income has a much larger scale. Without normalization, income would dominate the optimization, and age would contribute almost nothing. Normalization puts all features on equal footing.

### Classification Evaluation Metrics

Accuracy is the proportion of correct predictions. It seems like the obvious metric. But consider a dataset where 95% of the examples are "no" and only 5% are "yes." A classifier that always predicts "no" achieves 95% accuracy. Is it useful? Not at all.

The **confusion matrix** provides a complete breakdown:

|              | Predicted Positive | Predicted Negative |
|--------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

From these four numbers, we derive several metrics:

- **Precision** = TP / (TP + FP). Of the points you labeled as positive, how many were actually positive?
- **Recall** = TP / (TP + FN). Of the actual positive points, how many did you catch?
- **F1 score** = $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$. The harmonic mean of precision and recall. Use this when you care about both.

Imagine a confusion matrix displayed as a 2×2 heatmap. The diagonal cells (TP and TN) are correct predictions. The off-diagonal cells (FP and FN) are errors. Scanning this matrix tells you immediately where your model succeeds and where it fails.

## Methods Summary

| Feature | Simple Linear | Multiple Linear | Logistic |
|---------|---------------|-----------------|----------|
| Task | Regression | Regression | Binary Classification |
| Output | Continuous ($-\infty$ to $\infty$) | Continuous ($-\infty$ to $\infty$) | Probability (0 to 1) |
| Model Form | $y = mx + b$ | $y = \beta_0 + \sum \beta_i x_i$ | $P(y=1) = \text{sigmoid}(\beta_0 + \sum \beta_i x_i)$ |
| Objective | SSE | SSE | Cross-entropy loss |
| Evaluation | $R^2$, MSE, RMSE | $R^2$, MSE, RMSE | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Solution | Closed-form or numeric | Closed-form or numeric | Numerical optimization only |

-

All three regression methods carry assumptions. Violate them, and your results may still be useful, but you must proceed with caution.

**Linear regression** (simple and multiple) assumes:

- **Linearity**: The relationship between predictors and the target is linear.
- **Independence of errors**: Errors are not correlated with each other.
- **Homoscedasticity**: The variance of errors is constant across all levels of the predictors.
- **Normality of residuals**: The errors are normally distributed.

When these assumptions are violated, coefficient estimates may remain unbiased, but standard errors and p-values become unreliable. Confidence intervals may be too wide or too narrow. Hypothesis tests may be invalid.

**Logistic regression** assumes that the log-odds of the outcome are linear in the features. This is a less restrictive assumption than linear regression's normality assumption, but it is still consequential.

**Overfitting** becomes more severe as the number of features increases relative to the sample size. With many features, a linear model can achieve perfect training $R^2$ by memorizing the noise in the training set. This is the **curse of dimensionality**: as the feature space expands, the data becomes sparse, and models struggle to generalize.

**$R^2$ has notable limitations.** It always increases when you add features, even if those features are pure noise. Adjusted $R^2$ penalizes additional features, but negative validation $R^2$ remains the most honest diagnostic.

**Accuracy similarly misleads** on imbalanced datasets. A classifier that always predicts the majority class can achieve 95% accuracy while being completely useless. Precision, recall, and the F1 score provide a more complete picture.

Finally, **data leakage**—inadvertently allowing information from the validation or test sets to influence training—invalidates your entire evaluation. Common sources of leakage include:

- Scaling features before splitting the data
- Performing feature selection on the full dataset
- Using future information in time-series forecasts

The train-validation-test split is your first line of defense against leakage. **Never let the validation or test sets touch your training process.**

-

Simple linear, multiple linear, and logistic regression form a conceptual progression. All learn parameters by minimizing an objective function on training data. Simple and multiple linear regression minimize SSE and predict continuous outcomes. Logistic regression minimizes cross-entropy loss and predicts binary probabilities.

Proper evaluation requires separate training, validation, and test sets. Training metrics alone are dangerously optimistic—they hide overfitting. Overfitting manifests as good training performance but poor validation performance, often producing negative $R^2$ for regression or near-random accuracy for classification.

The confusion matrix and its derived metrics—precision, recall, and F1—provide deeper insight than accuracy alone. Accuracy can mislead, especially on imbalanced datasets.

Understanding the assumptions, limitations, and evaluation strategies of these foundational methods prepares the ground for more advanced machine learning techniques. Decision trees, random forests, support vector machines, and neural networks all build on the concepts introduced here.