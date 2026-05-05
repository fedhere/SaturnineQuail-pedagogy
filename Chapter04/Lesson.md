# Lesson 4: Null Hypothesis Rejection Testing

## Questions to Ponder


- The principle of falsifiability?
- What is Null Hypothesis Rejection Testing (NHRT)?
- Why do we use NHRT?
- What is the Z-test?
- What is a one-tailed test? What is a two-tailed test?
- What is a p-value?
- What is a statistic?
- What is p-hacking?
- What is the KS-Test?

## Introduction

The most basic formalism of science can be hastily distilled into one sentence: test an idea with data. That sentence does a lot of heavy lifting; testing an idea with data is hard! One question naturally arises: How do we know when data supports an idea and when it does not? This is where **Null Hypothesis Rejection Testing (NHRT)** comes in.

NHRT is based on the **Principle of Falsifiability** formulated by philosopher Karl Popper in 1934 in _The Logic of Scientific Discovery_.  An idea — formally called a **hypothesis** — must be falsifiable. Whatever your hypothesis is, it must be logically _and practically_ possible to contradict it. The canonical example of a falsifiable hypothesis is: "All swans are white." Simply observing one swan that is not white is enough to falsify this hypothesis.

## The NHRT Algorithm

Let's construct an example to demonstrate how we test hypotheses. Consider Bus who takes some route, Route $A$. The total trip duration of Route $A$ is known to follow a gaussian distribution with a mean of $\mu = 34 \text{ min}$ and a standard deviation of $\sigma=2.4 \text{ min}$. One day, the Bus company alters the bus route in order to shorten travel times. Let's call the new bus route Route $B$. The bus company measured the total trip duration for Route $B$ 100 times. Does this $N=100$ sample provide enough evidence to conclude that the Route $B$ is faster than Route $A$?

### Step 1: Formulate The Prediction

We start by constructing a **null hypothesis** ($H_0$). This must be a falsifiable statement that represents something we seek to falsify. A null hypothesis that states "The world is not round," would imply that our belief is that the world _is_ round and that we seek to show evidence for it. "The coin is fair," would imply that we are trying to show that the coin is not fair. For our bus example we might say that "Route $B$ produces the same trip durations as Route $A$." But we can get more specific.
 
<!---FBB dont use the word prove here! we speccifically do not want to prove, we want to falisify-->

> $H_0$: "The sample of Route $B$ trip durations comes from a population with $\mu_0 = 34 \text{ min}$.

Our goal with NHRT is to analyse our evidence and determine if we can *reject* the premise set forth by the null hypothesis.

### Step 2: Identify All Alternative Outcomes

Next we construct the **alternative hypothesis** ($H_1$), which is the logical complement of the null hypothesis. If $H_0$ is "The world is not round," then $H_1$ would be "The world is round." If $H_0$ is "The coin is fair," then $H_1$ would be "The coin is not fair." Notice how the combination of $H_0$ and $H_1$ encompasses every single logical possibility (conversely, e.g., "The world is not flat" would not complement "The world is not round," correctly as it leaves out infinite other possible shapes). In our case, the appropriate alternative hypothesis would be "Route $B$ produces different trip durations than Route $A$."

There is a small hiccup here. We are actually interested in knowing whether Route $B$ produces *shorter* trip durations than Route $A$. What we're interested in the **single-tailed** alternative (as opposed to a **double tailed** one where Route $B$ is simply a different duration, longer or shorter):

> $H_1$: "The sample of Route $B$ trip durations comes from a population with $\mu_0 < 34 \text{ min}$."

This choice of directionality has important consequences for the test, as we will see.

### Step 3: Set a Confidence Threshold

The scientist must determine how much evidence they require in order to reject the null hypothesis. We set this threshold of evidence by stating a confidence level, $\alpha$, which represents the probability of rejecting $H_0$ *when it's actually true.* This is known as a Type I error. Colloquially you could say "There is an $\alpha$\%  probability that the evidence indicates we should reject the null hypothesis when it's actually true." <!--The smallest value for $\alpha$ one should choose is $5$\%.-->

Once more: we must determine for ourselves how much evidence we need in order to reject the null hypothesis. No one else will do it for us. The way we do this is by selecting a confidence threshold *before* we perform the rest of the NHRT algorithm. This threshold represents the probability that the data indicates to us (through the use of the forthcoming statistical test) that we should reject the null hypothesis even though, in reality, it's true.

For the null hypothesis of "The world is round," we might choose $\alpha=0.01$. This means we are asserting that there is a $1$\% chance that the world is indeed round even though our data suggests otherwise. How can we possibly know what value of $\alpha$ to use? We don't. We don't set the confidence threshold by estimating it, we set it by *deciding* how rigorous we want our test to be. Choosing a very small threshold and then rejecting the null hypothesis indicates high confidence in the outcome of the test. You can set a large threshold, but the validity of your result will be less potent. There are standards on this in different fields. For example, in the social sciences, it is common to choose a $\alpha=0.05$ (5\%), since social sciences deal with human behavior, and humans are complex and erratic. In particle physics, conversely, the standard is $3\times10^{-7}$ (0.0000003), or about 1 in 3.5 million.

Let's choose $\alpha=0.05$ for this example.


### Step 4: Find a Pivotal Quantity with a Known Distribution Under $H_0$

A **pivotal quantity** (or test statistic) is a function of the data whose probability distribution is known when $H_0$ is true. For our example, we will choose the **Z-statistic**:

$$Z = \frac{\overline{X} - \mu_0}{\sigma_0 / \sqrt{n}}$$

where $\overline{X}$ is the sample mean, $n$ is the sample size and $\mu_0$ and $\sigma_0$ are the _population_ mean and standard deviation under $H_0$ (*i.e.,* $\mu_0=34\text{ min}$ $\sigma_0=2.4\text{ min}$. By the Central Limit Theorem, we know that when $n$ is sufficiently large, the sampling distribution of $\overline{X}$ is approximately normal, and consequently $Z$ follows a standard normal distribution with mean 0 and standard deviation 1, or in mathematical notation, $Z \sim P_\mathcal{N}$.

But is $Z$ is a number extracted from a standard normal distribution, from the previous lecture, we _know_ what the probability of getting a number _as large_ or _as small_ as $Z$ is because we know the PMF $P_\mathcal{N}$. 

For example, if $Z$, extracted from a Gaussian distribution with mean $\mu=0$ and standard deviation $\sigma=1$, were $Z>1$, we would know that the probability of getting $Z>1$ is $P(Z>1) = (100-68)/2 = 16$\% (please review last lecture's material if this is not clear)!


### Step 5: Calculate the Pivotal Quantity

Imagine that we calculate $\bar{X}$, which is the average of the 100 Route $B$ trip durations and find that $\bar{X}=34.46\text{ min}$. We then compute the Z-statistic:

$$Z = \frac{34.46 - 34}{2.4 / \sqrt{100}} = 1.94$$

The Z-statistic tells us that the sample mean of Route $B$ trip durations lies 1.94 standard deviations *above* the population mean of $\mu_0$. That is to say that the Route $B$ trip durations are longer than the Route $A$ trip durations. Not good. But let's see if the statistics say this is significant compared to our confidence threshold.

### Step 6: Test the Data Against Alternative Outcomes

The **p-value** is the probability, under the null hypothesis, of obtaining a test statistic at least as extreme as the one observed: $P(Z>1.94) if $H_0$ is true, or equivalently, if $Z \sim P_\mathcal{N}$. Crucially, the p-value is *not* the probability that $H_0$ is true.

$$p = 2 \Phi(| Z |) = 0.052$$

I've introduced a new symbol here, $\Phi$, the **Cumulative Distribution Function (CDF)** of the standard normal distribution. I'll come back to what the CDF is and how we calculated the $p\text{-value}$, but let's first accept this value as correct and figure out what it means for _null_ hypothesis.

Let's remind ourselves what our null and alternative hypotheses were. $H_0$: "The sample of Route $B$ trip durations comes from a population with $\mu_0 = 34 \text{ min}$. $H_1$: "The sample of Route $B$ trip durations comes from a population with $\mu_0 < 34 \text{ min}$."

The p-value is saying: "There is a $5.2$\% probability that one would observe our sample of Route $B$ trip durations while $H_0$ is true." In other words, this p-value is telling us that there is a $5.2$\% probability of observing these trip durations if Route $B$ is the same as Route $A$. This doesn't address whether Route $B$ is shorter or longer, just whether it's different.

We chose a p-value threshold of $5$\%, and our calculated p-value is *not* smaller than our threshold. This means that we *cannot* reject the null hypothesis.

There are two misconceptions we should talk about. First, the p-value is not the probability that the null hypothesis is true. It is the probability of observing data at least as extreme as what we did observe while assuming the null is true. Second, failing to reject $H_0$ is not the same as accepting $H_0$. Absence of evidence is not evidence of absence; our sample may be too small to detect a small effect.

#### The Cumulative Distribution Function

The CDF is the integral of the PDF along the support or, in simpler terms, the CDF is the area beneath the PDF curve. So if you imagine the bell shaped curve of the standard normal distribution, the CDF at some point $x$, $\Phi(x)$, is the area between the bell curve and the $x$-axis between $-\infty$ and $x$. In integral form, it looks like this:

$$\Phi(x) = \int_{-\infty}^{x} f(x') dx'$$

where $f(x)$ is the PDF and $x'$ is a dummy variable for integration.

What does the CDF tell us? The CDF evaluated at some point, $x$, is the probability that our random process will take on a value that is less than or equal to $x$. Relating this back to the bus routes example, we can learn some things. We already know that our $Z\text{-statistic}$ is distributed according to a standard normal distribution which, by definition, has mean of $0$. We calculated that $Z=1.94$. What is the probability that $Z$ could be either $\geq 1.94$ or $\leq -1.94$? In other words, what is the probability that $Z$ is at least as extreme as what we calculated. This is the $p\text{-value}$!
