# %% [markdown]
# ## Homework Three: Extending the Linear Regression Framework
# 
# In this homework, we will extend our understanding of linear regression by considering how using a **held-out test set** can help determine how well the model 
# generalizes to unseen data. Next week, we will refine this approach further by distinguishing between validation and testing sets.
# 
# Secondly, we will extend the linear regression framework by generalizing the basic idea of polynomial regression: we will perform **feature engineering** by
# adding new features computed from the originals using log scaling and other non-linear functions. 
# 
# **Notes:**
# 
# - **You will NO LONGER need to submit a `requirements.txt` file!**
# 
# - **Grading:** There are nine (9) graded questions, each worth 6 points. You will be given 2 bonus points for free to bring this up to a total of 50 points for the whole assignment. 

# %%
# Useful imports and utilities

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import kagglehub
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing,make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from tqdm import tqdm
from math import isclose

# globals

random_state = 42



# %% [markdown]
# ### Problem One:  Model Selection using Test Sets
# 
# This week’s video explores the challenge of model selection in polynomial regression: how do we determine the best polynomial degree for our data? As we saw, increasing model complexity (in this case, polynomial degree) generally reduces training MSE. However, this does not necessarily mean better generalization, so: 
# 
# - How do we select a model that performs well on new data?
# - How do we strike the right balance between:
#    - Underfitting (high bias, low variance), and
#    - Overfitting (low bias, high variance)?
# 
# To address this, we introduce **testing sets** and examine how evaluating models on unseen data provides a more reliable measure of generalization. 

# %% [markdown]
# ### Part A
# 
# The following function will generate datasets suitable for polynomial regression with various characteristics:
# 
# - `n_samples`: number of samples
# - `degree`: degree of the underlying polynomial model (with no noise)
# - `noise`: standard deviation of the noise added to the model
# - `show_data`, `show_model`: whether to show a plot of the dataset and the underlying model
# - `random_state`: random seed which controls how random choices are made; set to None if you want a different set of choices each time
# 
# Your first task, as usual, is to play around with this a bit, changing the parameters in the call in the last line. There is no coding to be done (yet). 

# %%
def make_poly_model(n_samples=10, degree=4,noise=1.0, show_data=False, show_model=True,random_state=random_state):
    
    lb,ub = 1,6

    def f(x):
        if degree==2:
            return 0.125*x**2 - 0.95*x + 1.925 
        elif degree==3:
            return 0.125*x**3 - 1.2*x**2 + 3*x + 0.5   
        elif degree==4:
            return 0.41*x**4 - 5.99*x**3 + 30.55*x**2 - 62.37*x + 41.4
        elif degree==5:
            return  -0.4*x**5 + 7.0*x**4 - 46.17*x**3 + 142.0*x**2 - 200.43*x + 102.0 
        elif degree==6:
            return -0.5219*x**6 + 10.7724*x**5 - 87.6206*x**4 + 355.7914*x**3 - 751.5246*x**2 + 774.0203*x - 300.917
        else:
            print("Degree must be in range [2..6]")
            return None
            
    X_all = np.linspace(lb,ub,1000)
    y_perfect_model = f(X_all)

    np.random.seed(random_state)
    X = np.linspace(lb,ub,n_samples)             # evenly spaced samples for simplicity
    y = f(X) + np.random.normal(0, noise,size=n_samples)

    # Plotting the scatter plot of the data 

    if show_data:
      
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, marker='.',label=f'Observed Data; Noise std = {noise}')
        if show_model:
            plt.plot(X_all, y_perfect_model, label='Underlying Model without noise', linestyle='--',color='grey', alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Degree {degree} Polynomial Noisy Dataset')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return X.reshape(-1,1),y

    
X,y = make_poly_model(n_samples=200,degree=5,noise=2,show_data=True,random_state=None)

# %% [markdown]
# ### Part B: Create a dataset with the following parameters:
# 
#     n_samples=200
#     degree = 5
#     noise=10
#     test_size = 0.3
#     random_state = 42
# 
# Leave the others at the defaults and then split it into four subsets using 
# sklearn's `test_train_split`; your results should match the test case. 

# %%

# Your code here

# Test: Should print (140, 1) (60, 1) (140,) (60,)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% [markdown]
# ### Part C: Investigating the model selection problem using test/train plots
# 
# In this part, you will complete the template below to:
# - Train polynomial models of degree 1 .. 10 on the training dataset just produced;
# - Calculate the training MSE and the testing MSE for each model;
# - Plot the MSE curves vs the model degree (see the Appendix for an example plot); and
# - Print out the minimal testing MSE found, and the "best" model which produced this result. 
# 
# In the remaining parts of the problem, you will answer questions about this plot. 
# 

# %%
# Template for Problem One Part C

degrees = range(1,11) # try polynomials of these degrees
train_errors = []     # store training MSEs for each degree here
test_errors  = []     # store testing MSEs for each degree here

for d in degrees:

    pass             # remove: just to get the cell to run without error

    # Use PolynomialFeatures to transform the data into appropriate form for
    # a polynomial model of degree d.

        # Your code here

    # Fit a linear regression model on the transformed data

        # Your code here

    # Predict on the train and test sets

        # Your code here

    # Calculate the MSEs and append to the appropriate lists

        # Your code here

    
# Plot training and test MSEs vs degree of model using train_errors and test_errors
# Note that the x axis should have ticks for 1 .. 10 only

# Your code here

# Calculate minimal MSE and degree for test set and print out 
# Hint: notice carefully that the MSE lists are for degrees [1, 2, 3, ....]  (i.e., index + 1)

# Your code here


# %% [markdown]
# ### Part D: Minimum MSE 
# 
# Set the variable to the minimum test MSE found for the given parameter choices.

# %%

a1d = 0.0                              # Just to get the cell to run without errors, replace by appropriate expression

print(f'a1d = {a1d:.4f}')              # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# ### Part E: Lower Bound for Testing MSE
# 
# The lower bound for the testing MSE is the MSE of the data points compared to
# the underlying model with no noise. 
# 
# Calculate this lower bound and assign it to the variable `a1e`. 
# 
# Hint: Construct a second dataset exactly the same as the original but with noise = 0.0.

# %%
# Your code here

a1e = 0.0                                  # Just to get the cell to run without errors, replace by appropriate expression

print(f'a1e = {a1e:.4f}')                  # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# #### For the last two parts of this problem, you must experiment with changes to the following parameters:
# 
# - `n_samples`:  Try 10, 20, 50, 100, 200, 500, 1000
# - `noise`: Try 0, 5, 10, 20, 50, 100, 500, 1000
# - `random_state`: None           (then you can try multiple times with the same parameter choices)
# 
# It is sufficient for this problem to simply change each of these
# separately, giving 7 + 8  = 15 trials. 
# 
# **Optional:**  For each parameter choice, try multiple times, with the random choices being different each time, by setting `random_state = None`. 
# 
# **Answer the following questions** based on these experiments. 
# 
# 

# %% [markdown]
# ### Part F: Training MSE
# 
# Set the variable to the single **most correct** answer.
# 
# As the degree increases, throughout all the experiments, the training MSE appears to ...
# 
# 1. Fluctuate unpredictably.
# 2. Never increase (gets smaller or stays the same in each step).
# 3. Decrease rapidly to degree 5 and then flatten out (not change much).
# 4. Always be higher than the test MSE.

# %%

a1f = ...                     # Must be one of 1, 2, 3, 4

print(f'a1f = {a1f}')         # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# ### Part G: Effect of Noise
# 
# Set the variable to the single **most correct** answer.
# 
# If we leave all parameters the same, except we set noise to increasingly larger values ...
# 
# 1. The best degree alternates between 5 and 6. 
# 2. The shapes of both plots is always essentially the same, the only change being the scale of the Y axis. 
# 3. The least MSE increases approximately proportionally to the square of the noise. 
# 4. The best degree found is always the same, and the least MSE increases approximately proportionally with the noise. 

# %%
# Give the degree where this minimum test MSE was found

a1g = ...                              # Must be one of 1, 2, 3, 4

print(f'a1g = {a1g}')                  # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# ### Optional: Run more tests, with different degrees of polynomial as well!
# 
# If you have time to try more experiments, say by changing the degree, you will see that I chose the default parameters for our investigations
# with some care, because you won't always get the beautiful curves shown in the testbooks, and you won't always get the correct degree when selecting
# the "best" model based on the testing MSE. 

# %% [markdown]
# ## **Intro to Problem Two**
# 
# We have seen that we can preprocess data by constructing new features that encode **nonlinear functions** of existing features. In statistical learning theory, these transformations are often called **basis functions**. So far, we have explored only the simplest case—extending a univariate regression problem with polynomial terms like $x^2$, $x^3$, and so on. However, **any** nonlinear transformation of the features can be computed and either **added to** or **used in place of** existing features.
# 
# #### **Examples of Nonlinear Basis Functions**
# #### **1. Polynomial Transforms**
# So far, we have only considered the univariate case of polynomial regression, where we add new features such as $x^2, x^3, \dots$. However, with multiple features, we also generate **interaction terms** such as $x_1 x_2$.  
# 
# For example, a full quadratic polynomial regression on features $x_1$ and $x_2$ introduces three additional terms:
# 
# $$
# y = \beta_0
# + \beta_1 x_1 + \beta_2 x_2
# + \underline{\beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2}
# $$
# 
# 
# 
# Since the number of terms in polynomial expansions grows very quickly, it is common to limit interactions to only a few key features.
# 
# #### **2. Log Transforms**
# A logarithmic transformation—such as $\log(y)$—is useful when a feature (or the target variable) spans multiple orders of magnitude or appears to grow exponentially fast. This transformation helps stabilize variance, making relationships more linear and easier to model.
# 
# #### **3. Exponential Transforms**
# In contrast to log transforms, an exponential transformation** (e.g., $2^{x_1}$) can be useful when a feature exhibits exponential decay instead of growth. This can help capture relationships where small changes in a feature have large effects on the target variable.
# 
# 

# %% [markdown]
# ## Problem Two: Linear Regression with Feature Engineering on the California Housing Dataset
# 
# The transforms we will investigate in this problem, using the **CA Housing Dataset**, are as follows:
# 
# #### 2.1 Log-transform the target:
# 
# Because housing prices can vary wildly,  replace $y$ by its log-transform:
# 
# $$
#     y = \log(\text{MedHouseVal}).
# $$
# 
# 
# #### 2.2 Log-transform a feature
# 
# 
# Since  *Median Income* ranges widely, add a feature
# 
# $$
#     \text{LogMedInc} = \log(\text{MedInc}).
# $$
# 
# 
# #### 2.3 Polynomial features for slight curvature
# 
# You might suspect that increasing *population* from 1,000 to 2,000 has a bigger effect on price than from 10,000 to 11,000 (diminishing returns).  
# Thus, add a quadratic term to include polynomial regression on this feature:
# 
# $$
#     \text{PopSquared} = (\text{Population})**2
# $$
# 
# 
# #### 2.4 Interaction features
# 
# It might be that *average rooms* matter more if *average bedrooms* is above some threshold. One way to let your linear model handle this is to add the product term:
# 
# $$\text{RoomsTimesBeds} = \text{AveRooms} \times \text{AveBedrms}. $$
# 
# In an ordinary linear regression, you’d then have something like
# 
# $$\text{MedHouseVal} \approx \beta_0 + \beta_1\,\text{AveRooms} + \beta_2\,\text{AveBedrms} + \beta_3\,\text{RoomsTimesBeds} + \dots$$
# 
# The coefficient $\beta_3$ shows how the slope for *AveRooms* depends on *AveBedrms* (and vice versa).
# 
# 
# 
# **Note:** The usual practice is that we **replace** the target by a new target, but **add** transformed features. In a later lesson, we will investigate **feature selection**, in which we may delete features which are insignificant or noisy. 
# 
# 

# %% [markdown]
# ## Your Turn!
# In this problem, you will apply feature engineering to the **California Housing Dataset** and evaluate its impact on linear regression performance.
# 
# ### Part A. Load the Data  
# - Code to load the dataset from `sklearn.datasets` is provided.  
# 
# ### Part B. Train Regression Models & Compute Metrics  
# Perform linear (multiple) regression using `MedHouseVal` as the target and compute **testing MSE and $R^2$ values** for each of the following cases:  
# 
# 1. **Baseline Model** – Use the original dataset without feature engineering.  
# 2. **Single Transformations Models** – Apply each of the transformations (2.1–2.4) one at a time, keeping all other features unchanged (resulting in 4 models).  
# 3. **All Transformations Model** – Apply all transformations simultaneously and evaluate the model.  
# 
# ### Part C. Summarize & Visualize Results  
# - Display the results in **tabular format** showing testing MSE and $R^2$ values for all experiments (don't overthink this, but at least label and line up each column).  
# - Plot results using two **bar charts** (at least give a title to each chart and label each bar).  
#   - Compare the MSEs but exclude models with log-transformed target (where the MSE is in log-space and not directly comparable).  
#   - Compare $R^2$ values for all models. 
# 
# ### Part D. Complete Graded Questions  
# 
#  
# **Note: To ensure reproducibility, always use `random_state=42` in all relevant functions.**
# 

# %% [markdown]
# ### Part A

# %%
# 1. Load the California Housing data into a DataFrame
data = fetch_california_housing(as_frame=True)
df = data.frame  # The dataset in pandas DataFrame format

df.head()

# %% [markdown]
# ### Part B
# 
# You may create additional code cells as you wish. 

# %%
# Your code here

# %% [markdown]
# ### Part C

# %%
# Your code here

# %% [markdown]
# ### Part D: Graded answers
# 

# %%
# Assign to the variable the MSE of the original data set with no transformations

a2d1 = 0.0                              # Just to get the cell to run without errors, replace by appropriate expression

print(f'a2d1 = {a2d1:.4f}')             # Do NOT change this print statement in any way, it is used by the auto-grader!

# %%
# Assign to the variable the MSE  score of the dataset after transformation 2.3

a2d2 = 0.0                              # Just to get the cell to run without errors, replace by appropriate expression

print(f'a2d2 = {a2d2:.4f}')             # Do NOT change this print statement in any way, it is used by the auto-grader!

# %%
# Assign to the variable the $R^2$  score of the dataset after transformation 2.1


a2d3 = 0.0                              # Just to get the cell to run without errors, replace by appropriate expression

print(f'a2d3 = {a2d3:.4f}')             # Do NOT change this print statement in any way, it is used by the auto-grader!

# %%
# Assign to the variable the R^2 score of the best result, i.e., the best R^2 score found in any experiment
# Note that you do not have to specify which experiment(s), just give the largest value.

a2d4 = 0.0                              # Just to get the cell to run without errors, replace by appropriate expression

print(f'a2d4 = {a2d4:.4f}')             # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# ### Part E

# %%
# How many hours did you spend on this homework?  Assign to the variable an integer value. 

a2e = ...                         # Just to get the cell to run without errors, replace by appropriate expression

print(f'a2e = {a2e}')             # Do NOT change this print statement in any way, it is used by the auto-grader!

# %% [markdown]
# ## Appendix
# 
# Here is a sample of the graphic I expect in Problem One. 
# 
# ![Screenshot 2025-01-31 at 8.30.09 AM.png](attachment:81ef3298-d253-4747-ac07-ffb5bb545e66.png)


