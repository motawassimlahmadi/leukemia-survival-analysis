import polars as pl
import numpy as np

def replace_energized(mood):
    """
    Replaces "Energize" by a random value
    """
    return np.random.choice(["Fatigued", "Neutral"]) if mood == "Energized" else mood

def replace_fatigued(mood):
    """
    Replaces "Fatigued" by a random value
    """
    return np.random.choice(["Energized", "Neutral"]) if mood == "Fatigued" else mood

def replace_neutral(mood):
    """
    Replaces "Neutral" by a random value
    """
    return np.random.choice(["Energized", "Fatigued"]) if mood == "Neutral" else mood

def replace_stressed(_):
    """
    Replaces "Stressed" by a random value
    """
    return np.random.choice(["Energized", "Fatigued", "Neutral"])



def apply_mood_transformations(df):
    """
    Handling incoherent values in mood after workout 
    """
    # 1) HIIT , Strength and Energized
    df = df.with_columns(
        pl.when(
            (pl.col("Workout Type") == "HIIT") | (pl.col("Workout Type") == "Strength") &
            (pl.col("Workout Intensity") == "High")
        )
        .then(pl.col("Mood After Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 2) Calories burned and intensive training
    df = df.with_columns(
        pl.when(
            ((pl.col("Workout Type") == "Cardio") | (pl.col("Workout Type") == "HIIT") |
             (pl.col("Workout Type") == "Running") | (pl.col("Workout Type") == "Cycling")) &
            (pl.col("Workout Intensity") == "High") & (pl.col("Calories Burned") > 700)
        )
        .then(pl.col("Mood After Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 3) Little sleep and intensive training
    df = df.with_columns(
        pl.when(
            (pl.col("Sleep Hours") < 5) &
            (pl.col("Workout Intensity") == "High")
        )
        .then(pl.col("Mood After Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 4) Yoga and low intensity
    df = df.with_columns(
        pl.when(
            (pl.col("Workout Type") == "Yoga") &
            (pl.col("Workout Intensity") == "Low")
        )
        .then(pl.col("Mood After Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 5) Happy before workout and low intensity
    df = df.with_columns(
        pl.when(
            (pl.col("Mood Before Workout") == "Happy") &
            (pl.col("Workout Intensity") == "Low")
        )
        .then(pl.col("Mood After Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 6) High heart rate and intensity
    df = df.with_columns(
        pl.when(
            (pl.col("Heart Rate (bpm)") >= 170) &
            (pl.col("Workout Intensity") == "High")
        )
        .then(pl.col("Mood After Workout").map_elements(replace_neutral, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    # 7) Yoga and lots of sleep
    df = df.with_columns(
        pl.when(
            (pl.col("Workout Type") == "Yoga") &
            (pl.col("Sleep Hours") >= 8)
        )
        .then(pl.col("Mood After Workout").map_elements(replace_fatigued, return_dtype=pl.String))
        .otherwise(pl.col("Mood After Workout"))
        .alias("Mood After Workout")
    )

    return df

def transform_mood_before_workout(df):
    """
    Changing vocabulary of mood before workout to match mood after workout
    """
    # Mapping
    mapping = {
        "Tired": "Fatigued",
        "Happy": "Energized",
    }

    df = df.with_columns(
        pl.col("Mood Before Workout").replace(mapping)
    )

    # Replacing Stressed
    df = df.with_columns(
        pl.when(
            (pl.col("Mood Before Workout") == "Stressed")
        )
        .then(pl.col("Mood Before Workout").map_elements(replace_stressed, return_dtype=pl.String))
        .otherwise(pl.col("Mood Before Workout"))
        .alias("Mood Before Workout")
    )

    # Incoherent to be energized with less than 6 hours of sleep
    df = df.with_columns(
        pl.when(
            (pl.col("Sleep Hours") < 6)
        )
        .then(pl.col("Mood Before Workout").map_elements(replace_energized, return_dtype=pl.String))
        .otherwise(pl.col("Mood Before Workout"))
        .alias("Mood Before Workout")
    )

    return df

def add_derived_features(df):
    """
    Add derived features 
    """
    df = df.with_columns(
        (pl.col("Distance (km)") * 1000).alias("Distance (km)"),  # Conversion en mètres
        ((pl.col("Weight (kg)") * 30)/1000).alias("Water Intake (liters)"),  # Quantité d'eau nécessaire
        ((pl.col("Sleep Hours")*60).alias("Sleep Hours"))  # Conversion en minutes
    )

    # Features engineering
    df = df.with_columns(
        ((pl.col("Calories Burned") / pl.col("Workout Duration (mins)"))).alias("CB / Duration"),
        (pl.when(pl.col("Sleep Hours") >= 7).then(1).otherwise(0).alias("Quality of Sleep"))
    )

    return df

def encode_categorical_features(df):
    """
    Encoding 
    """
    # Columns to encode
    col_encode = ["Gender", "Workout Type", "Workout Intensity"]

    for col in col_encode:
        dic = {v: k for k, v in enumerate(df[col].unique())}
        df = df.with_columns(
            pl.col(col).replace_strict(dic)
        )

    # Replacing neutral mood
    df = df.with_columns(
        pl.col("Mood After Workout").map_elements(replace_neutral, return_dtype=pl.Utf8).alias("Mood After Workout"),
        pl.col("Mood Before Workout").map_elements(replace_neutral, return_dtype=pl.Utf8).alias("Mood Before Workout")
    )

    col_mapping = ["Mood Before Workout", "Mood After Workout"]
    mapping_mood = {
        "Fatigued": 0,
        "Energized": 1
    }

    for col in col_mapping:
        df = df.with_columns(
            pl.col(col).replace_strict(mapping_mood)
        )

    return df

def handle_missing_values(df):
    """
    Imputation of missing values
    """
    df = df.with_columns(**{
        col: pl.col(col).fill_null(strategy='mean') for col in [
            "Steps Taken", "Distance (km)", "Sleep Hours", "Daily Calories Intake",
            "Age", "Height (m)", "Workout Duration (mins)", "Heart Rate (bpm)",
            "Resting Heart Rate (bpm)", "Body Fat (%)", "Workout Type",
            "Calories Burned", "Water Intake (liters)", "Weight (kg)", "IMC"
        ]
    })

    return df

def create_age_bins(df):
    """
    Binned age
    """
    df = df.with_columns(
        pl.when((pl.col("Age") >= 18) & (pl.col("Age") <= 28)).then(0)
        .when((pl.col("Age") > 28) & (pl.col("Age") <= 38)).then(1)
        .otherwise(2)
        .alias("Binned Age")
    )

    return df

def normalize_features(df):
    """
    Min Max normalization
    """
    col_to_normalize = ["Calories Burned", "Distance (km)", "Steps Taken", "Daily Calories Intake" , "CB / Duration" , "Weight (kg)" , "Heart Rate (bpm)" , "Resting Heart Rate (bpm)",
                        "IMC" , "Body Fat (%)"  , "Workout Duration (mins)"]

    for colmn in col_to_normalize:
        df = df.with_columns(
            ((pl.col(colmn) - pl.col(colmn).min()) / (pl.col(colmn).max() - pl.col(colmn).min())).alias(colmn)
        )

    return df

def drop_unnecessary_columns(df):
    """
    Drops unnecessary columns
    """
    return df.drop("Age" , "Sleep Hours")

def build_features(df):
    """
    Applying features functions
    """
    df = apply_mood_transformations(df)
    df = transform_mood_before_workout(df)
    df = encode_categorical_features(df)
    df = handle_missing_values(df)
    df = add_derived_features(df)
    df = create_age_bins(df)
    df = normalize_features(df)
    df = drop_unnecessary_columns(df)

    return df