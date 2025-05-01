import polars as pl
import numpy as np
import random

def clean_gender(df):
    """
    Cleans the Gender column
    """
    return df.with_columns(
        pl.col("Gender").replace("Other", None)
    ).fill_null(strategy="backward")



def calculate_body_fat(df):
    """
    Calculates body fat percentage for men and women
    """
    # Calculation for men
    bdy_fat_male = (
        df.select(["User ID", "Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat (%)"])
        .filter(pl.col("Gender") == "Male")
    )

    # Size conversion in meters
    bdy_fat_male = bdy_fat_male.with_columns(
        (pl.col("Height (cm)") / 100).alias("Height (m)")
    )

    # Calculating BMI
    bdy_fat_male = bdy_fat_male.with_columns(
        (pl.col("Weight (kg)") / pl.col("Height (m)")**2).alias("IMC")
    )

    # Calculating body fat percentage
    bdy_fat_male = bdy_fat_male.with_columns(
        (
            (pl.col("IMC") * 1.2) + (0.23 * pl.col("Age")) - 16.2
        ).alias("Body Fat (%)")
    )

    # Calculation for women
    bdy_fat_female = (
        df.select(["User ID", "Age", "Gender", "Height (cm)", "Weight (kg)", "Body Fat (%)"])
        .filter(pl.col("Gender") == "Female")
    )

    # Size conversion in meters for women
    bdy_fat_female = bdy_fat_female.with_columns(
        (pl.col("Height (cm)") / 100).alias("Height (m)")
    )

    # Calculating BMI
    bdy_fat_female = bdy_fat_female.with_columns(
        (pl.col("Weight (kg)") / pl.col("Height (m)")**2).alias("IMC")
    )

    # Calculating body fat percentage for women
    bdy_fat_female = bdy_fat_female.with_columns(
        (
            (pl.col("IMC") * 1.2) + (0.23 * pl.col("Age")) - 5.4
        ).alias("Body Fat (%)")
    )

    # Joining the two dataframes
    union_df = pl.concat([bdy_fat_male, bdy_fat_female])

    # Joining with the original dataframe
    result_df = df.join(union_df, on="User ID", how="left")

    # Deleting the duplicated columns
    col_to_drop = ["Age_right", "Gender_right", "Height (cm)_right", "Weight (kg)_right", "Body Fat (%)", "Height (cm)"]

    return result_df.drop(col_to_drop)

def merge_datasets(df, df1):
    """
    Merges the two datasets
    """
    # Cleaning df1
    df1 = df1.drop(["Max_BPM", "Workout_Frequency (days/week)", "Experience_Level", "BMI"])

    # Adding USER_ID in df1
    df1 = df1.with_columns(
        pl.arange(10001, 10001 + df1.height).alias("User ID")
    )

    # Merging
    df_merged = df.join(df1, on="User ID", how="full")

    # Replacing null values
    df_merged = df_merged.with_columns(
        pl.coalesce(["User ID", "User ID_right"]).alias("User ID"),
        pl.coalesce(["Gender", "Gender_right"]).alias("Gender"),
        pl.coalesce(["Weight (kg)", "Weight (kg)_right"]).alias("Weight (kg)"),
        pl.coalesce(["Heart Rate (bpm)", "Avg_BPM"]).alias("Heart Rate (bpm)"),
        pl.coalesce(["Body Fat (%)_right", "Fat_Percentage"]).alias("Body Fat (%)"),
        pl.coalesce(["Calories Burned", "Calories_Burned"]).alias("Calories Burned"),
        pl.coalesce(["Workout Type", "Workout_Type"]).alias("Workout Type"),
        pl.coalesce(["Age", "Age_right"]).alias("Age"),
        pl.coalesce(["Height (m)", "Height (m)_right"]).alias("Height (m)"),
        pl.coalesce(["Workout Duration (mins)", "Session_Duration (hours)"]).alias("Workout Duration (mins)"),
        pl.coalesce(["Water Intake (liters)", "Water_Intake (liters)"]).alias("Water Intake (liters)"),
        pl.coalesce(["Resting Heart Rate (bpm)", "Resting_BPM"]).alias("Resting Heart Rate (bpm)")
    )

    # Deleting duplicated columns
    col_to_drop = [
        "Age_right", "Gender_right", "Weight (kg)_right", "Avg_BPM",
        "Resting_BPM", "Session_Duration (hours)", "Calories_Burned", "Workout_Type",
        "Fat_Percentage", "Water_Intake (liters)", "User ID_right", "Resting_BPM", "Body Fat (%)_right", "Height (m)_right"
    ]

    return df_merged.drop(col_to_drop)

def clean_workout_type(df):
    """
    Ceaning workout type 
    """
    mapping = {
        '\\tCardio': "Cardio",
        "\\nStrength": "Strength",
        "\\tYoga": "Yoga"
    }

    return df.with_columns(
        pl.col("Workout Type").replace(mapping)
    )

def handle_mood_columns(df):
    """
    Handle null mood values
    """
    # Mood After Workout
    col = "Mood After Workout"
    col_val = df[col]

    moods = ["Fatigued", "Energized", "Neutral"]

    # Null values
    for index, elem in enumerate(col_val):
        if elem is None:
            col_val[index] = random.choice(moods)

    df = df.rename({"Mood After Workout": "Mood after workout"})
    df = df.insert_column(18, col_val)
    df = df.drop("Mood after workout")

    # Mood Before Workout
    col = "Mood Before Workout"
    col_val = df[col]

    moods = ["Tired", "Stressed", "Happy", "Neutral"]

    for index, elem in enumerate(col_val):
        if elem is None:
            col_val[index] = random.choice(moods)

    df = df.rename({"Mood Before Workout": "Mood before workout"})
    df = df.insert_column(17, col_val)
    df = df.drop("Mood before workout")

    return df

def filter_outliers(df):
    """
    Filtering the outliers
    """
    df = df.filter(pl.col("Body Fat (%)") >= 5)  # Valeur aberrante : bodyfat <= 4 %
    df = df.filter(pl.col("Body Fat (%)") <= 48)  # Valeur aberrante : bodyfat >= 48%

    return df

def preprocess_data(df, df1=None):
    """
    Principal preprocessing function
    """
    df = clean_gender(df)
    df = calculate_body_fat(df)

    if df1 is not None:
        df = merge_datasets(df, df1)

    df = clean_workout_type(df)
    df = df.drop("User ID", "VO2 Max")
    df = handle_mood_columns(df)
    df = filter_outliers(df)

    return df