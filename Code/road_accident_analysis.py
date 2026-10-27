# @title
# ---------------------------------------------------------
# ROAD ACCIDENT DATA ANALYSIS (with simple explanations)
# ---------------------------------------------------------

# Import the tools (libraries) we need. The short names are nicknames.
import pandas as pd              # pandas  -> works with tables
import numpy as np               # numpy   -> does maths (mean, median, etc.)
import matplotlib.pyplot as plt  # matplotlib -> draws charts
import seaborn as sns            # seaborn -> makes charts look nicer

# Make the console show all columns (otherwise long tables get cut with ...)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------
# LOAD THE DATA
# ---------------------------------------------------------
# Read the CSV file into a table called df (DataFrame)
df = pd.read_csv("data.csv")

print("ROAD ACCIDENT DATA ANALYSIS")
print("\nOriginal Dataset:")
print(df.head())     # show the first 5 rows

# ---------------------------------------------------------
# STEP 1: DATA CLEANING (Pandas)
# ---------------------------------------------------------
# Count empty (missing) cells in every column. We want 0.
print("\nMissing Values:")
print(df.isnull().sum())

# Remove repeated rows, if any
df = df.drop_duplicates()

# List of the 5 number columns we will analyse
columns = [
    "Accidents", "Fatalities", "Injuries",
    "Avg_Speed_kmph", "Traffic_Volume"
]

# If any cell is empty, fill it with the median (middle value) of its column.
# Our data has no empty cells, so nothing changes. This is only for safety.
for col in columns:
    df[col] = df[col].fillna(df[col].median())

print("\nCleaned Dataset:")
print(df.head())

# New column: deaths per 100 accidents (shows how deadly accidents are)
df["Fatality_Rate"] = (df["Fatalities"] / df["Accidents"] * 100).round(2)

# New column: road type. "Highway A" becomes "Highway" (removes the last letter)
df["Road_Type"] = df["Road_Location"].str.replace(r"\s[A-D]$", "", regex=True)

# ---------------------------------------------------------
# STEP 2: STATISTICS (NumPy)
# ---------------------------------------------------------
print("\nStatistical Analysis:")
for col in columns:                       # repeat for each column
    values = df[col].to_numpy()           # turn the column into a NumPy array
    print("\nParameter:", col)
    print("Mean:", round(np.mean(values), 2))        # average
    print("Median:", round(np.median(values), 2))    # middle value
    print("Minimum:", np.min(values))                # smallest value
    print("Maximum:", np.max(values))                # largest value
    print("Total:", np.sum(values))                  # sum of all values
    print("Std Deviation:", round(np.std(values), 2))  # how spread out values are

# Average of every column for each road type (Highway, Urban Road, etc.)
print("\nAverage by Road Type:")
print(df.groupby("Road_Type")[columns + ["Fatality_Rate"]].mean().round(2))

# Correlation: how strongly two columns move together (-1 to +1)
# +1 = both go up together, -1 = one goes up while the other goes down
print("\nCorrelation Matrix:")
print(df[columns].corr().round(2))

# Show risky roads: fatality rate above 25 OR speed above 60 km/h
print("\nHigh-risk samples (Fatality_Rate > 25 or Avg_Speed_kmph > 60):")
print(df[(df["Fatality_Rate"] > 25) | (df["Avg_Speed_kmph"] > 60)]
      [["Road_Location", "Fatality_Rate", "Avg_Speed_kmph"]])

# ---------------------------------------------------------
# STEP 3: CHARTS (Matplotlib and Seaborn)
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")   # clean white background with grid lines

# Graph 1: Bar chart - accidents at each location
plt.figure(figsize=(9, 5))                       # size of the chart
sns.barplot(data=df, x="Road_Location", y="Accidents")
plt.title("Number of Accidents by Road Location")
plt.xticks(rotation=45)                          # tilt names so they don't overlap
plt.tight_layout()                               # fit everything neatly
plt.show()                                       # display the chart

# Graph 2: Histogram - how many locations fall in each fatalities range
plt.figure(figsize=(8, 5))
plt.hist(df["Fatalities"], bins=5, edgecolor="black")   # bins=5 -> 5 groups
plt.title("Distribution of Fatalities")
plt.xlabel("Fatalities")
plt.ylabel("Number of Locations")
plt.tight_layout()
plt.show()

# Graph 3: Scatter plot - each dot is one road; colour shows road type
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Avg_Speed_kmph", y="Fatalities", hue="Road_Type")
plt.title("Average Speed vs Fatalities")
plt.tight_layout()
plt.show()

# Graph 4: Heatmap - correlation table drawn in colours
# red = strong positive link, blue = strong negative link
plt.figure(figsize=(8, 5))
sns.heatmap(
    df[columns].corr(),
    annot=True,        # write the numbers inside the boxes
    cmap="coolwarm",   # blue-to-red colours
    fmt=".2f"          # show 2 decimal places
)
plt.title("Road Accident Correlation")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# SAVE RESULT
# ---------------------------------------------------------
# Save the cleaned table (with the 2 new columns) as a new file
df.to_csv("cleaned_data.csv", index=False)
print("\nAnalysis completed successfully!")
print("Cleaned data saved to cleaned_data.csv")