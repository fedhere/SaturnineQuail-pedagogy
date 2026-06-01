# Lesson 5: Introduction to Machine Learning

## Questions to Ponder

- What are the four levels of measurement, and why do they matter?
- What is machine learning, and how is it different from traditional programming?
- What is a model? What is the difference between a parameter and a hyperparameter?
- What is the distinction between supervised and unsupervised learning?
- How do objective functions (L1, L2, and others) guide model fitting?
- What is Ordinary Least Squares (OLS) regression, and why does it have an analytic solution?
- How do we evaluate model performance using metrics like MSE, RMSE, and $R^2$?
- Why do we split data into training, validation, and test sets?
- What are overfitting and underfitting, and how do we detect them?

## Introduction

Imagine you are trying to teach a computer to recognize handwritten digits. You could try to write explicit rules: "A '7' has a horizontal line across the top and a diagonal line down to the right." But what about different handwriting styles? What about smudges? What about a '7' written with a serif? You would spend forever writing rules, and your program would still fail on the first handwritten note you gave it.

There must be a better way.

There is. Instead of programming the rules explicitly, you can show the computer many examples of handwritten digits and let it discover the rules itself. This is **machine learning**.

## Part 1: What Is Data?

Before we can build models, we must understand the types of data we work with. The **NOIR taxonomy** classifies data into four levels of measurement. Each level supports different mathematical operations, and using the wrong operation on the wrong type of data leads to nonsense.

| Level | Order | Distance | Mean | Median | Mode | Absolute Zero |
|-------|-------|----------|------|--------|------|----------------|
| Nominal | ✓ | | | | ✓ | |
| Ordinal | ✓ | | | ✓ | ✓ | |
| Interval | ✓ | ✓ | ✓ | ✓ | ✓ | |
| Ratio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Nominal data** consist of unordered categories. Colors are nominal. Bus route numbers are nominal. You can say one color is not "more" than another. You cannot calculate a mean. The mode—the most frequently occurring category—is the only appropriate measure of central tendency.

**Ordinal data** have order but not equal spacing. Movie ratings (1–5 stars) are ordinal. You know that 5 stars is better than 1 star, but you cannot claim that a 4-star movie is "twice as good" as a 2-star movie. The difference between 4 stars and 5 stars may not mean the same thing as the difference between 1 star and 2 stars. The median and mode are appropriate; the mean is not.

**Interval data** have equal spacing but no true zero. Temperature in Celsius or Fahrenheit is interval. Twenty degrees Celsius is hotter than ten degrees Celsius, and the ten-degree difference means the same thing anywhere on the scale. However, twenty degrees Celsius is *not* twice as hot as ten degrees Celsius because zero degrees Celsius does not represent an absence of temperature. The mean, median, and mode are all meaningful.

**Ratio data** have equal spacing *and* a true zero. Temperature in Kelvin is ratio. Twenty Kelvin *is* twice as hot as ten Kelvin. Zero Kelvin represents absolute zero—the complete absence of thermal energy. All arithmetic operations, including ratios, are valid.

**NB:** Many introductory data science projects treat all numeric data as ratio data. This is a mistake. If your data lacks a true zero, reporting that "the average temperature doubled" is mathematically incorrect.

## Part 2: What Is Machine Learning?

Arthur Samuel, a pioneer of artificial intelligence, defined machine learning in 1959 as "the field of study that gives computers the ability to learn without being explicitly programmed." Traditional programming requires you to provide explicit rules. Machine learning requires you to provide examples. The computer discovers the rules from the examples.

A **model** is a low-dimensional representation of a higher-dimensional dataset. Consider a scatter plot of 100 points that roughly follow a straight line. The raw data contains 100 x-values and 100 y-values—200 numbers. But the linear trend can be described by just two numbers: the slope and the intercept. That is a model. It is a simplification that captures the essential pattern while discarding the noise.

**Parameters** are learned from the data. In the linear model $y = mx + b$, the slope $m$ and intercept $b$ are parameters. The learning algorithm adjusts them to fit the training data.

**Hyperparameters** are set by the user before training begins. How many neighbors should a k-nearest neighbors model consider? How deep should a decision tree grow? These choices are hyperparameters. They are not learned from the data; you must choose them based on experience, intuition, or validation performance.

### Supervised vs. Unsupervised Learning

In **unsupervised learning**, all features are observed for all data points. You have no "right answer" to check against. You are simply looking for structure within the feature space. Common applications include:

- **Clustering**: Partitioning data into groups of similar points
- **Anomaly detection**: Identifying unusual observations that do not fit the pattern
- **Dimensionality reduction**: Compressing data while preserving important information

In **supervised learning**, some features (called targets or labels) are unobserved for some data points. You want to predict them. The data points for which the target *is* observed constitute your labeled training set. Common applications include:

- **Classification**: Predicting discrete categories (spam or not spam? cat or dog?)
- **Regression**: Predicting continuous values (house price, temperature tomorrow)

**Semi-supervised learning** combines a small amount of labeled data with a large amount of unlabeled data. This is common when labeling is expensive but unlabeled data is abundant.

**Active learning** allows the model to interactively query the user for labels on particularly informative data points. The model asks: "What is *this* one?" and you tell it.

## Part 3: Model Fitting and Objective Functions

Fitting a model involves two steps.

**Step 1:** Choose a mathematical form—a family of functions—that represents the expected relationship in the data. For a linear relationship, you might choose $y = mx + b$.

**Step 2:** Minimize an **objective function** (also called a loss function) that quantifies how poorly the model fits the data. The objective function takes the model's predictions, compares them to the true values, and returns a single number: the error.

The **L1 objective** (Sum of Absolute Errors) is:

$$\sum |y_i - (mx_i + b)|$$

L1 is robust to outliers—a single extreme point does not dramatically change the optimal parameters. However, L1 can produce multiple optimal solutions, which can be confusing.

The **L2 objective** (Sum of Squared Errors) is:

$$\sum (y_i - (mx_i + b))^2$$

L2 penalizes large errors more heavily than L1. If your prediction is off by 10, L2 counts that as 100—ten times worse than being off by 1 (which contributes 1). L2 has a unique analytic solution, which makes it fast and predictable.

Other objective functions include **Huber loss** (which combines the robustness of L1 with the smoothness of L2), **cross-entropy loss** (used for classification), and **hinge loss** (used for support vector machines).

### Ordinary Least Squares (OLS) Regression

For the linear model $y_i = \alpha + \beta x_i$, minimizing the L2 objective yields the Ordinary Least Squares solution. Remarkably, this problem has an **analytic solution**—you can compute the optimal parameters directly using formulas, without iterative searching.

$$\hat{\beta} = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$

$$\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$$

The slope $\hat{\beta}$ captures the expected change in $y$ for a one-unit change in $x$. The intercept $\hat{\alpha}$ is the expected value of $y$ when $x = 0$.

I encourage you to derive these formulas yourself. Start with the L2 objective, take the derivative with respect to $\beta$ and $\alpha$, set both derivatives to zero, and solve. It is a satisfying exercise.

## Part 4: Training, Validation, and Testing

Here is a trap that catches many newcomers. You fit a model to your data. You evaluate its performance on the *same* data. The performance looks great—$R^2 = 0.95$! You celebrate. Then you try the model on new data, and it fails catastrophically.

What happened? Your model memorized the training data, including its noise and quirks. It did not learn the underlying pattern. This is **overfitting**.

To honestly assess generalization, you must split your dataset into three non-overlapping subsets before doing anything else.

**The training set** (typically 60-80% of your data) is where the model learns its parameters. The model sees this data and adjusts its coefficients to minimize the objective function.

**The validation set** (typically 10-20%) is used to tune hyperparameters and compare different models. You evaluate performance here repeatedly, but you do *not* train on it.

**The testing set** (typically 10-20%) is used exactly once, at the very end of your project, to report final performance. You never look at the testing set until you are done making decisions.

### Model Performance Metrics

For regression problems (predicting continuous values), common metrics include:

$$AE = \sum_i |\epsilon_i| \quad \text{(Absolute Error)}$$
$$SE = \sum_i \epsilon_i^2 \quad \text{(Squared Error)}$$
$$MSE = \frac{1}{N} SE \quad \text{(Mean Squared Error)}$$
$$RMSE = \sqrt{MSE} \quad \text{(Root Mean Squared Error)}$$
$$R^2 = 1 - \frac{MSE}{\sigma^2} \quad \text{(Coefficient of Determination)}$$

The $R^2$ value represents the proportion of variance in the target explained by the model. It ranges from $-\infty$ to 1.

- $R^2 = 1$: Perfect prediction. Your model explains all the variance.
- $R^2 = 0$: Your model performs no better than simply predicting the mean.
- $R^2 < 0$: Your model performs *worse* than predicting the mean. This is a clear sign of severe overfitting or a misspecified model.

For classification problems (predicting discrete categories), common metrics include:

- **Accuracy**: The proportion of correct predictions
- **Precision**: True positives divided by predicted positives. Of the points you labeled as positive, how many were actually positive?
- **Recall**: True positives divided by actual positives. Of the actual positive points, how many did you catch?
- **F1 score**: The harmonic mean of precision and recall. This balances the two when you care about both.

### Overfitting and Underfitting

**Underfitting** occurs when your model is too simple to capture the underlying pattern. Imagine fitting a straight line to data that clearly follows a curve. Underfitting produces poor performance on both the training set and the test set.

**Overfitting** occurs when your model is too complex relative to your amount of data. The model learns noise and idiosyncrasies specific to the training set. Overfitting produces excellent performance on the training set but poor performance on the test set.

The train-validation-test split is your primary defense against overfitting. If validation performance lags noticeably behind training performance, you are overfitting. Your options include: simplifying the model, collecting more data, adding regularization, or reducing the number of features.

**NB:** The concepts in this lesson—particularly the NOIR taxonomy, the distinction between supervised and unsupervised learning, and the train-validation-test split—will appear throughout the rest of this course. Get comfortable with them now, and the later lessons will come much more easily.