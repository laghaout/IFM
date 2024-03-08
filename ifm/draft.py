# create a sample DataFrame
df = pd.DataFrame(
    {
        "gender": [
            "male",
            "female",
            "male",
            "female",
            "male",
            "male",
            "female",
        ],
        "education": [
            "high school",
            "college",
            "college",
            "graduate",
            "high school",
            "graduate",
            "college",
        ],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000],
    }
)

# convert categorical variables to numeric
df["gender"] = df["gender"].astype("category").cat.codes
df["education"] = df["education"].astype("category").cat.codes
print(df)
