# %% [markdown]
# # Homework 04:  Null Values, Categorical Features, and Cross Validation
# 
# In this homework, we are going to add three **tools to your toolbox** which will be essential when you work with real datasets:
# 1. What do we do with null-values?
# 2. How do we deal with non-numeric features?
# 3. What validation strategy provides the best estimate of the final testing score?
# 
# For (1), we'll explore several ways of dealing with null values:
# - Removing columns with too many null values,
# - Imputing values for missing categorical labels using the "most frequent" category strategy, and
# - Imputing values for missing numeric values using the median. 
# 
# 
# For (2), we'll use ordinal encoding to replace categorical labels with floats.
# 
# For (3), we'll try three different cross-validation strategies:
# 
# - 5-Fold CV,
# - Repeated 5-Fold CV, and
# - Leave one out CV, 
# 
# and see which comes closest to estimating the final testing MSE. 
# 
# 
# #### Grading: There are eight (8) answers to provide, each worth 6 points.  (You get 2 points for free.)
#  

# %%
# Useful imports

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, LeaveOneOut
from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import OrdinalEncoder, OneHotEncoder  
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import mean_squared_error, r2_score
from tqdm                    import tqdm



# %% [markdown]
# ### Load the Ames Housing Dataset
# 
# For a description of the features of this dataset, see the **Appendix**. 

# %%
# Download the latest version of the dataset
path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")

# print("Path to dataset files:", path)

# Construct the full path to the CSV file (update the file name if necessary)
csv_file = os.path.join(path, "AmesHousing.csv")

# Read the dataset into a DataFrame
df = pd.read_csv(csv_file)

# %%
# Print the first few rows of the DataFrame
df.head()

# %%
# Uncomment this to see the listing of features

# df.info()

# %%
# Uncomment this to see the feature histograms

# print("Feature Histograms")
# df.hist(figsize=(15, 13), bins=30)  # Adjust figure size and number of bins
# plt.tight_layout()  # Adjust spacing to prevent overlap
# plt.show()

# %% [markdown]
# ### Data Preprocessing
# 
# First, let's remove the features that are clearly not useful for regression.

# %%
df_clean = df.drop(columns=['Order','PID'])
df_clean.head()

# %% [markdown]
# ### Problem One: Dealing with Null Values
# 
# There are basically two strategies for dealing with missing (null or `NaN`) values:
# - Get them out of your dataset by **removing** features and/or samples containing too many nulls.
# - **Impute** values by replacing nulls with the mean, median, or other "neutral" value computed from the feature.
# 
# **Note:** It is also possible to impute values using more advanced techniques such as mode imputation, forward/backward fill, or predictive modeling (e.g., KNN or regression-based imputation). These techniques might be useful when you start to work on your projecct. 

# %% [markdown]
# **First we will explore how many null values occur in each feature.**

# %%
# This function will list how many nulls occur in which features

def show_null_counts_features(df):
    # Count the nulls and calculate the %
    count_nulls = df.isnull().sum()
    df_nulls = (df.isnull().mean() * 100).round(2)
    
    # Determine if the column is numeric or non-numeric
    feature_types = df.dtypes.apply(lambda x: 'Numeric' if np.issubdtype(x, np.number) else 'Categorical')
    
    # Filter out the columns with missing values and sort them in descending order
    missing_data = pd.DataFrame({
        'Feature': count_nulls[count_nulls > 0].index,
        '# Null Values': count_nulls[count_nulls > 0].values, 
        'Null %': df_nulls[df_nulls > 0].values,
        'Type': feature_types[count_nulls > 0].values
    }).sort_values(by='Null %', ascending=False)
    
    print(f'The dataset contains {len(df)} samples.\n')

    if (len(missing_data) == 0):
        print("There are no null values in the dataset!")
    else:
        # Print null value stats
        print('Feature Name    # Nulls      Null %    Type')
        print('------------    -------      ------    ----')
        for index, row in missing_data.iterrows():
            print(f"{row['Feature']:<15} {row['# Null Values']:<12} {row['Null %']:.2f}%   {row['Type']}")

show_null_counts_features(df_clean)

# %% [markdown]
# ### Part A
# 
# Clearly, some of these features are not very informative! Let's drop the worst offenders!
# 
# **Fill in your code after the comments below to drop any features with more than `max_nulls` null values.**
# 

# %%
max_nulls = 500      # We will drop any features with more than max_nulls missing values

# Count null values per column

# -> Your code

# Filter out columns where null count exceeds max_nulls

# -> Your code

# Drop the columns

# -> Your code

# Uncomment to verify they were removed

# show_null_counts_features(df_clean)

# %%
# Set this variable to the number of cells that were dropped

a1a = ...

print(f"a1a = {a1a}")                                   # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ### Part B:  Feature Transformations for Imputing Null Values
# 
# Now let's perform the following feature transformations:
# 
# - For categorical features, we'll replace null values with the most frequent category in that column
# - For numeric features, we'll replace nulls with the median for that column
# 
# 
# This is very simple to do with a couple of lines of Python, but naturally we want to use `sklearn` functions whenever we can, so we'll use ` SimpleImputer`.
# 
# **Go read the doc page for `SimpleImputer` before proceeding.**

# %% [markdown]
# ### Imputing Categorical Features using the Most Frequent Strategy

# %% [markdown]
# In the next cell you see how easy it is to examine the categories. **Take a moment and explore several of the categorical features.**  In this dataset, most of them are skewed, with a clear "most favorite" category. 
# (If the feature values are not skewed, then you could change these to a new category "Unknown".)

# %%
df_clean['Garage Qual'].value_counts(dropna=False)

# %% [markdown]
# Before proceeding, let's get lists of the two types of features.

# %%
# Identify categorical and numeric features

categorical_features = df_clean.select_dtypes(exclude=['number']).columns.tolist()
numeric_features     = df_clean.select_dtypes(include=['number']).columns.tolist()

# Print results if you want
# print("Categorical Features:", categorical_features)
# print("Numeric Features:", numeric_features)

# %% [markdown]
# **Now you must impute all the categorical features using `SimpleImputer` with the `most_frequent` strategy.**

# %%
# First make a copy of the cleaned dataset, call it df_imputed

# -> Your code

# Impute categorical columns (using most frequent category)

# -> Your code

# Verify: only numeric features should appear

# show_null_counts_features(df_imputed)

# %%
# Set this variable to the number of occurrences of the category 'TA' in the feature 'Garage Qual'
# It should have increased from before the imputation, because Nan values were changed to 'TA'

a1b = ...     

print(f"a1b = {a1b}")                                   # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ### Part C:  Imputing Numeric Features using the Median
# 
# Now you must "simply impute" values for the numeric features using the `median` strategy. 

# %%
# Impute numeric columns (using the median)

# -> Your code

# Verify: There should be no null values

# show_null_counts_features(df_imputed)

# %%
# Nothing to do here:   Answer should be 0 

a1c = df_imputed.isnull().any(axis=1).sum()      # count number of rows with any missing values

print(f"a1c = {a1c}")                            # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ### Part D:   Ordinal encoding the categorical features with OrdinalEncoder
# 
# The simplest option in dealing with categorical values is to represent them by integers 0, 1, 2, etc.
# 
# **Before proceeding, read the doc page on `sklearn`'s `OrdinalEncoder`.**
# 
# Follow the comments to perform this feature transformation

# %%
# Put df_imputed in the form X, y

# -> Your code

# Initialize OrdinalEncoder

# -> Your code

# Fit and transform categorical columns

# -> Your code

# Convert back to DataFrame to retain column names 

# -> Your code

# Verify

# X.head()

# %%
# Nothing to do here:   Answer should show categories encoded as floats for 'Lot Shape'

a1d = X['Lot Shape'].unique()                      

print(f"a1d = {a1d}")                            # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ## Problem Two:  Train and Test a Regression Model with Cross-Validation
# 
# In this problem, we will perform a regression on the Ames Housing Dataset using several different cross-validation
# strategies, comparing the cross-validation score for each with the final testing MSE, to see which provides the best
# estimate of the final test score, and hence of the model's ability to generalize. 
# 
# 
# We shall compare each of the following cross-validation MSEs with the final test MSE score:
# 
# - 5-Fold Cross-Validation (default)
# - Repeated 5-Fold Cross-Validation (repeated 100 times)
# - Leave-One-Out Cross Validation
# 
# **Note:  Set `n_jobs = -1` when doing cross validation to take advantage of parallelism in your environment.** 

# %% [markdown]
# ### Part A: 5-Fold Cross-Validation
# 
# For this part
# - Create a train-test split with `test_size=0.2` 
# - Create a linear model and perform K-fold cross-validation with K = 5 and using `cross_val_score` with `scoring='neg_mean_squared_error'` (remember to take the mean of the CV scores and negate the result, since scoring uses a negative MSE). 
# - Report (print out) the
#     - CV score (negated mean of MSE measurements over all K folds)
#     - Test MSE
# 
# 
# Use `random_state = 42` for all experiments. 

# %%
# Your code here

# -> Your code

# %%
# Assign to this variable the Test MSE 

a2a = ...               

print(f"a2a = {a2a:.4f}")                            # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ## Part B: Perform Repeated 5-Fold Cross Validation
# 
# Read the doc page on `sklearn`'s `RepeatedKFold` for cross validation; repeat the CV calculation with K = 5 and `n_repeats=100`
# and report the CV score (negated mean of MSE measurements over all 100*K folds)

# %%
# Your code here

# -> Your code

# %%
# Assign to this variable the mean CV score for the repeated K-fold experiment 
# Note: if your CV score is negative, go back and read the instructions for Part A again!

a2b = ...               

print(f"a2b = {a2b:.4f}")                            # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ## Part C: Perform Leave One Out Cross Validation
# 
# This is simply a matter of setting `cv=LeaveOneOut()`. Run the same experiment and report the CV score. 

# %%
# Your code here

# -> Your code

# %%
# Assign to this variable the mean CV score for the leave-one-out experiment 

a2c = ...               

print(f"a2c = {a2c:.4f}")                            # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ### Part D
# 
# Now, in order to help interpret the results, print out a table of the **square roots** of each of the CV scores and the final test score; we can then see the result in the same units (dollars) as the target, instead of the units of the MSE (dollars squared). 
# 
# Hint: Here is an example of how to print out values as currency:
# 
#     cost = 23512.23
#     print(f"cost: ${cost:,.2f}")

# %%
# Your code here 

# -> Your code

# %%
# Assign to this variable the letter of the CV strategy whose RMSE came closest to the actual test score

a2d = ...                      # Should be 'A' = 5-fold CV; 'B' = Repeated 5-Fold CV; or 'C' = LOO CV                   

print(f"a2d = {a2d}")         # Don't delete or change this line, it is needed by the auto-grader

# %% [markdown]
# ### Optional
# 
# - Try K-Fold CV with various K
# - Try `RepeatedKFold` with various K and various `n_repeated`. 

# %% [markdown]
# ## Appendix: Explanation of Features in Ames Housing Dataset
# 
# ### **Identification**
# - `PID` → Parcel Identification Number (unique identifier for each property)
# - `Order` → Row number (used for indexing, not a feature)
# 
# ---
# 
# ### **Sale Information**
# - `SalePrice` → The final selling price of the house in USD (**Target variable**)
# - `Mo Sold` → Month the house was sold (1 = January, ..., 12 = December)
# - `Yr Sold` → Year the house was sold
# - `Sale Type` → Type of sale (e.g., **WD** = Warranty Deed, **New** = Newly Built)
# - `Sale Condition` → Condition of the sale (e.g., **Normal**, **Abnormal**, **Partial** for incomplete homes)
# 
# ---
# 
# ### **General Property Information**
# - `MS SubClass` → Type of dwelling (e.g., **20 = 1-story**, **60 = 2-story**, **120 = Townhouse**)
# - `MS Zoning` → Zoning classification (e.g., **RL = Residential Low Density**, **C = Commercial**)
# - `Lot Frontage` → Linear feet of street connected to property
# - `Lot Area` → Total size of the lot in square feet
# - `Neighborhood` → Physical locations within Ames (e.g., **CollgCr = College Creek**)
# - `Condition 1` / `Condition 2` → Proximity to roads or railroads (e.g., **Norm = Normal**, **PosN = Near Park**)
# 
# ---
# 
# ### **Building & House Design**
# - `Bldg Type` → Type of dwelling (e.g., **1Fam = Single Family**, **Twnhs = Townhouse**)
# - `House Style` → Style of the house (e.g., **1Story = One Story**, **2Story = Two Story**, **SplitFoyer**)
# - `Overall Qual` → Overall quality of materials (scale: **1 = Very Poor** to **10 = Excellent**)
# - `Overall Cond` → Overall condition of the house (scale: **1 = Very Poor** to **10 = Excellent**)
# 
# ---
# 
# ### **Year Built & Remodel**
# - `Year Built` → Original construction year
# - `Year Remod/Add` → Year of last remodel or addition
# 
# ---
# 
# ### **Exterior Features**
# - `Exterior 1st` / `Exterior 2nd` → Exterior covering material (e.g., **VinylSd = Vinyl Siding**, **HdBoard = Hardboard**)
# - `Mas Vnr Type` → Masonry veneer type (e.g., **BrkFace = Brick Face**, **None = No Veneer**)
# - `Mas Vnr Area` → Area of masonry veneer in square feet
# 
# ---
# 
# ### **Basement Features**
# - `Bsmt Qual` → Basement height (e.g., **Ex = Excellent**, **TA = Typical**, **Po = Poor**)
# - `Bsmt Cond` → General condition of the basement
# - `Bsmt Exposure` → Walkout or garden level basement?
# - `BsmtFin Type 1` / `BsmtFin SF 1` → Primary finished area in basement (e.g., **GLQ = Good Living Quarters**)
# - `BsmtFin Type 2` / `BsmtFin SF 2` → Secondary finished area
# - `Bsmt Unf SF` → Unfinished square feet in basement
# - `Total Bsmt SF` → Total square footage of basement
# 
# ---
# 
# ### **Utilities & HVAC**
# - `Heating` → Type of heating system (e.g., **GasA = Gas Forced Air**, **OthW = Hot Water Heating**)
# - `Heating QC` → Quality of heating system (e.g., **Ex = Excellent**, **Fa = Fair**)
# - `Central Air` → **Y = Yes**, **N = No**
# - `Electrical` → Electrical system (e.g., **SBrkr = Standard Breaker**, **FuseA = Fuse Box**)
# 
# ---
# 
# ### **Above Ground Living Area**
# - `1st Flr SF` → First-floor square footage
# - `2nd Flr SF` → Second-floor square footage
# - `Gr Liv Area` → Total **above-ground** living area in square feet
# - `Low Qual Fin SF` → Low-quality finished square feet (e.g., unfinished rooms)
# 
# ---
# 
# ### **Bathrooms & Bedrooms**
# - `Full Bath` → Full bathrooms above ground
# - `Half Bath` → Half bathrooms above ground
# - `Bsmt Full Bath` → Full bathrooms in basement
# - `Bsmt Half Bath` → Half bathrooms in basement
# - `Bedroom AbvGr` → Number of bedrooms above ground
# - `Kitchen AbvGr` → Number of kitchens above ground
# - `Kitchen Qual` → Kitchen quality (**Ex = Excellent**, **Fa = Fair**)
# 
# ---
# 
# ### **Garage Features**
# - `Garage Type` → Type of garage (e.g., **Attchd = Attached**, **Detchd = Detached**)
# - `Garage Yr Blt` → Year garage was built
# - `Garage Finish` → Interior finish of garage
# - `Garage Cars` → Size of garage in car capacity
# - `Garage Area` → Garage size in square feet
# 
# ---
# 
# ### **Additional Features**
# - `Fireplaces` → Number of fireplaces
# - `Fireplace Qu` → Fireplace quality
# - `Paved Drive` → Paved driveway? (**Y = Yes, P = Partial, N = No**)
# - `Wood Deck SF` → Square footage of wood deck
# - `Open Porch SF` → Square footage of open porch
# - `Enclosed Porch` → Square footage of enclosed porch
# - `Screen Porch` → Square footage of screened porch
# - `Pool Area` → Pool area in square feet
# - `Misc Val` → Miscellaneous features (e.g., shed value)
# 
# 


