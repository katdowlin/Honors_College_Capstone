# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:52:26 2025

@author: kevin
"""


from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf


def preprocess_data(filepath, rower_id, race_id):
    df = pd.read_csv(filepath, skiprows=28, header=None, engine='python')

    df.columns = df.iloc[0]             
    df = df[1:].reset_index(drop=True)  

    df = df.drop_duplicates(subset=["Total Strokes"], keep="first")
    df = df.drop(columns=[
        "Heart Rate", "Distance (IMP)", "Split (IMP)", "Speed (IMP)", "Distance/Stroke (IMP)",
        "GPS Lat.", "GPS Lon.", "Interval", "Elapsed Time", "Split (GPS)"
    ], errors='ignore')
    df = df.dropna()

    cols_to_convert = [
        "Speed (GPS)", "Stroke Rate", "Total Strokes", "Distance/Stroke (GPS)",
        "Power", "Catch", "Slip", "Finish", "Wash", "Force Avg", "Work",
        "Force Max", "Max Force Angle"
    ]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols_to_convert, how='all')
    df["rower"] = rower_id
    df["race"] = race_id
    return df


race_data = []

datasets = [
    # === Doc Hosea (Race 101) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-03-30 Doc Hosea\SpdCoach 2937082 20240330 0848AM.csv", 4, 1),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-03-30 Doc Hosea\SpdCoach 2937091 20240330 0848AM.csv", 3, 1),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-03-30 Doc Hosea\SpdCoach 2937094 20240330 0848AM.csv", 2, 1),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-03-30 Doc Hosea\SpdCoach 2937100 20240330 0848AM.csv", 1, 1),

    # === Ivy Invitational (Race 102) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-04-12 Ivy Invitational\SpdCoach 2937082 20240412 0123pm.csv", 3, 2),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-04-12 Ivy Invitational\SpdCoach 2937100 20240412 0124pm.csv", 1, 2),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-04-12 Ivy Invitational\SpdCoach 2937091 20240412 0124pm.csv", 4, 2),

    # === Big Ten Invitational 1 (Race 103) ===
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 1\SpdCoach 2937082 20240419 0840am.csv", 3, 3),
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 1\SpdCoach 2937091 20240419 0840am.csv", 4, 3),

    # === Big Ten Invitational 2 (Race 104) ===
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 2\SpdCoach 2937082 20240419 0337pm.csv", 3, 4),
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 2\SpdCoach 2937091 20240419 0423pm.csv", 4, 4),

    # === Big Ten Invitational 3 (Race 105) ===
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 3\SpdCoach 2937082 20240420 0848am.csv", 3, 5),
    (r"C:\Users\kevin\Downloads\2024-04-19 Big10 Invitational-20250328T173223Z-001\2024-04-19 Big10 Invitational\Big10 Invite Heat 3\SpdCoach 2937091 20240420 0848am.csv", 4, 5),

    # === Eastern Sprints Heats (Race 106) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Heats\SpdCoach 2937082 20240505 0810am.csv", 4, 6),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Heats\SpdCoach 2937091 20240505 0810am.csv", 3, 6),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Heats\SpdCoach 2937094 20240505 0810am.csv", 2, 6),

    # === Eastern Sprints Finals (Race 107) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Finals\SpdCoach 2937082 20240505 0303pm.csv", 4, 7),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Finals\SpdCoach 2937091 20240505 0303pm.csv", 3, 7),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-05 Eastern Sprints\Finals\SpdCoach 2937094 20240505 0303pm.csv", 2, 7),

    # === Big Ten Championships (Race 108) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-19 Big10 Championships\SpdCoach 2937082 20240519 1120am.csv", 4, 8),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-19 Big10 Championships\SpdCoach 2937091 20240519 1120am.csv", 3, 8),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-19 Big10 Championships\SpdCoach 2937094 20240519 1120am.csv", 2, 8),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-19 Big10 Championships\SpdCoach 2937100 20240519 1120am.csv", 1, 8),

    # === NCAA Heats (Race 109) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937094 20240531 1200pm.csv", 2, 9),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937091 20240531 1200pm.csv", 3, 9),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937082 20240531 1200pm.csv", 4, 9),


    # === NCAA Semis (Race 110) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937091 20240601 0932am.csv", 3, 10),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937082 20240601 0932am.csv", 4, 10),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937094 20240601 0932am.csv", 2, 10),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937100 20240601 0932am.csv", 1, 10),

    # === NCAA Finals (Race 111) ===
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937082 20240602 0850am.csv", 4, 11),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937091 20240602 0850am.csv", 3, 11),
    (r"C:\Users\kevin\Downloads\Empower Oarlock Project\2024-05-31 NCAA Championships-20250210T194216Z-001\2024-05-31 NCAA Championships\SpdCoach 2937100 20240602 0850am.csv", 1, 11),
]

for filepath, rower_id, race_id in datasets:
        cleaned = preprocess_data(filepath, rower_id, race_id)
        race_data.append(cleaned)


ALLMERGED = pd.concat(race_data, ignore_index=True)

ALLMERGED = ALLMERGED[(ALLMERGED["Total Strokes"] >= 100) & (ALLMERGED["Total Strokes"] <= 200)].copy()
ALLMERGED.sort_values(by=["race", "rower", "Total Strokes"], inplace=True)
ALLMERGED.reset_index(drop=True, inplace=True)

ALLMERGED["Effective Length"] = abs(
    (ALLMERGED["Catch"] + ALLMERGED["Slip"]) - (ALLMERGED["Finish"] - ALLMERGED["Wash"])
)

ALLMERGED.dropna(inplace=True)

ALLMERGED.to_csv("FourData_output.csv", index=False)


def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    initial_bearing_rad = math.atan2(x, y)
    initial_bearing_deg = (math.degrees(initial_bearing_rad) + 360) % 360

    return initial_bearing_deg


race_coords = {
    1: (39.9251497, -75.0828046, 39.9259478, -75.0592425),  # DOCHO
    2: (40.3503283, -74.6268953, 40.3681693, -74.6237977),  # IVY
    3: (27.3578881, -82.4536639, 27.3759389, -82.4537451), #BIGIN
    4: (27.3578881, -82.4536639, 27.3759389, -82.4537451),
    5: (27.3578881, -82.4536639, 27.3759389, -82.4537451),
    6:(42.2940421,-71.7573363,42.2759985,-71.7561799), #EAST
    7:(42.2940421,-71.7573363,42.2759985,-71.7561799),
    8: (43.4267561, -89.7331116, 43.4090164, -89.7289197), #BIGTEN
    9: (39.01753, -84.1575, 39.02037, -84.1325), #NCAA
    10: (39.01753, -84.1575, 39.02037, -84.1325),
    11: (39.01753, -84.1575, 39.02037, -84.1325),

}


race_headings = {
    race_id: calculate_bearing(lat1, lon1, lat2, lon2)
    for race_id, (lat1, lon1, lat2, lon2) in race_coords.items()
}


for race_id, heading in race_headings.items():
    print(f"Race {race_id} Heading: {heading:.2f}°")


cardinal_to_deg = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}


wind_heading_data = pd.DataFrame({
    'race': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Wind Speed': [13,14,3,13,3,9,9,6,8,3,0],
    'Wind Direction_Cardinal': ['NW','W','NW','W','N','ESE','SSE','NE', 'WSW', 'SSW', 'N'],
})

wind_heading_data['Wind Direction_Degrees'] = wind_heading_data['Wind Direction_Cardinal'].map(
    cardinal_to_deg)

wind_heading_data['Boat Heading'] = wind_heading_data['race'].map(race_headings)


def compute_wind(row):
    angle_diff = math.radians(
        row['Wind Direction_Degrees'] - row['Boat Heading'])
    return pd.Series({
        'Wind_Along': row['Wind Speed'] * math.cos(angle_diff),
        'Wind_Cross': row['Wind Speed'] * math.sin(angle_diff)
    })


wind_heading_data[['Wind_Along', 'Wind_Cross']
                  ] = wind_heading_data.apply(compute_wind, axis=1)


ALLMERGED = pd.merge(ALLMERGED, wind_heading_data[[
                     'race', 'Wind_Along', 'Wind_Cross']], on='race', how='left')


group_cols = ['race', 'Total Strokes']

variability = ALLMERGED.groupby(group_cols)[['Catch', 'Finish']].agg(['std', 'max', 'min'])

variability.columns = ['Catch_SD', 'Catch_Max', 'Catch_Min', 'Finish_SD', 'Finish_Max', 'Finish_Min']
variability = variability.reset_index()


crew_counts = ALLMERGED.groupby(['race', 'Total Strokes'])['rower'].count().reset_index()
crew_counts.rename(columns={'rower': 'CrewSize'}, inplace=True)

ALLMERGED_counts = pd.merge(ALLMERGED, crew_counts, on=['race', 'Total Strokes'], how='left')


variability = ALLMERGED.groupby(['race', 'Total Strokes'])[['Catch', 'Finish']].std().reset_index()
variability.rename(columns={'Catch': 'Catch_SD', 'Finish': 'Finish_SD'}, inplace=True)


ALLMERGED_var = pd.merge(ALLMERGED_counts, variability, on=['race', 'Total Strokes'], how='left')

ALLMERGED_var["Effective_Catch"] = ALLMERGED_var["Catch"] + ALLMERGED_var["Slip"]
ALLMERGED_var["Effective_Finish_SD"] = ALLMERGED_var["Finish"] - ALLMERGED_var["Wash"]


ALLMERGED_var["Effective_Catch"] = ALLMERGED_var["Catch"] + ALLMERGED_var["Slip"]
ALLMERGED_var["Effective_Finish_SD"] = ALLMERGED_var["Finish"] - ALLMERGED_var["Wash"]


sd_by_race = ALLMERGED_var.groupby(['race', 'Total Strokes'])[
    ['Effective_Catch', 'Effective_Finish_SD']
].std().reset_index()


sd_by_race.rename(columns={
    'Effective_Catch': 'Effective_Catch_SD',
    'Effective_Finish_SD': 'Effective_Finish_SD_SD'
}, inplace=True)


ALLMERGED_var = pd.merge(
    ALLMERGED_var,
    sd_by_race,
    on=['race', 'Total Strokes'],
    how='left'
)

ALLMERGED["Effective_Catch"] = ALLMERGED["Catch"] + ALLMERGED["Slip"]
ALLMERGED["Effective_Finish"] = ALLMERGED["Finish"] - ALLMERGED["Wash"]

boat_level_stats = ALLMERGED.groupby(["race", "Total Strokes"]).agg(
    Avg_Power_per_Stroke=("Power", "mean"),
    SD_Power=("Power", "std"),
    SD_Catch=("Catch", "std"),
    SD_Finish=("Finish", "std"),
    SD_Effective_Catch=("Effective_Catch", "std"),
    SD_Effective_Finish=("Effective_Finish", "std"),
    Avg_Speed=("Speed (GPS)", "mean")
).reset_index()

ALLMERGED_merged = pd.merge(ALLMERGED, boat_level_stats, on=["race", "Total Strokes"], how="left")

from sklearn.preprocessing import StandardScaler

avg_power_col = ["Avg_Power_per_Stroke"]
sd_cols = ["SD_Power", "SD_Effective_Catch", "SD_Effective_Finish"]


scaler_power = StandardScaler()
scaler_sd = StandardScaler()


ALLMERGED_merged["Avg_Power_z"] = scaler_power.fit_transform(ALLMERGED_merged[avg_power_col])


ALLMERGED_merged[[col + "_z" for col in sd_cols]] = scaler_sd.fit_transform(ALLMERGED_merged[sd_cols])


ALLMERGED_merged["PowerFit_Interaction1_z"] = ALLMERGED_merged["Avg_Power_z"] * ALLMERGED_merged["SD_Power_z"]
ALLMERGED_merged["PowerFit_Interaction2_z"] = ALLMERGED_merged["Avg_Power_z"] * ALLMERGED_merged["SD_Effective_Catch_z"]
ALLMERGED_merged["PowerFit_Interaction3_z"] = ALLMERGED_merged["Avg_Power_z"] * ALLMERGED_merged["SD_Effective_Finish_z"]


corr_cols = [
    "Speed (GPS)", "Avg_Power_per_Stroke", "SD_Power", "SD_Catch", "SD_Finish",
    "SD_Effective_Catch", "SD_Effective_Finish", "PowerFit_Interaction1_z",
    "PowerFit_Interaction2_z", "PowerFit_Interaction3_z"
]
correlations = ALLMERGED_merged[corr_cols].corr()

#ALLMERGED_var['Desync_Index'] = (ALLMERGED_var['Catch_SD'] + ALLMERGED_var['Finish_SD']) / np.sqrt(ALLMERGED_var['CrewSize'])

ALLMERGED_var['Desync_Index'] = (ALLMERGED_var['Catch_SD'] + ALLMERGED_var['Finish_SD'])


ALLMERGED_var = ALLMERGED_var.dropna()


#Models and Analysis

X = ALLMERGED[['Stroke Rate', 'Power', 'Catch', 'Slip', 'Finish', 'Wash',
                         'Force Avg', 'Work', 'Max Force Angle', 'Effective Length', 'Wind_Along', 'Wind_Cross']]

y = ALLMERGED['Speed (GPS)']

X_with_desync = ALLMERGED_var[['Stroke Rate', 'Power', 'Catch', 'Slip', 'Finish', 
                           'Wash', 'Force Avg', 'Work', 'Max Force Angle', 
                           'Effective Length', 'Wind_Along', 'Wind_Cross', 'Desync_Index']]

yv = ALLMERGED_var['Speed (GPS)']

X_trainv, X_testv, y_trainv, y_testv = train_test_split(X_with_desync, yv, test_size=0.2)

# Linear regression
lin_model = LinearRegression()
lin_model.fit(X_trainv, y_trainv)
lin_preds = lin_model.predict(X_testv)

print("Linear Model R² with Desync Index:", r2_score(y_testv, lin_preds))

X_with_dummies = pd.get_dummies(ALLMERGED[['Stroke Rate', 'Power', 'Catch', 'Slip', 'Finish', 
                                           'Wash', 'Force Avg', 'Work', 'Max Force Angle', 
                                           'Effective Length', 'Wind_Along', 'Wind_Cross', 'race']], 
                                columns=['race'], drop_first=True)


X_traind, X_testd, y_traind, y_testd = train_test_split(X_with_dummies, y, test_size=0.2)
model = LinearRegression()
model.fit(X_traind, y_traind)
preds = model.predict(X_testd)

print("R² with dummy race variables:", r2_score(y_testd, preds))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print('Mean Squared Error:', mse)
print('R2 Score regular:', r2)

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-Validated R² Scores:", scores)
print("Mean R² Cross_Valid:", scores.mean())


gkf = GroupKFold(n_splits=5)

sk_model = LinearRegression()

scores = cross_val_score(
    sk_model, X, y, 
    cv=gkf.split(X, y, groups=ALLMERGED['race']), 
    scoring='r2'
)

print("Grouped CV R² Scores:", scores)
print("Mean Grouped R²:", scores.mean())


X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()

print(model.summary())

num_merged = ALLMERGED_var

plt.figure(figsize=(12, 8))
sns.heatmap(num_merged.corr(), annot=True,
            cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Random Forest Feature Importances:")
print(importances.sort_values(ascending=False))

explainer = shap.Explainer(rf, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)


residuals = y_test - predictions

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs. Predicted")
plt.xlabel("Predicted Speed")
plt.ylabel("Residuals")
plt.show()


#mixed effects (race is random effect)

ALLMERGED_merged = ALLMERGED_merged.dropna(subset=[
    'Speed (GPS)', 'Stroke Rate', 'SD_Power', 'Effective Length',
    'Max Force Angle', 'PowerFit_Interaction1_z', 'SD_Effective_Catch', 'SD_Effective_Finish',
    'PowerFit_Interaction2_z', 'PowerFit_Interaction3_z', 'Avg_Power_per_Stroke'
])

ALLMERGED_merged = ALLMERGED_merged.reset_index(drop=True)


ALLMERGED_merged['race'] = ALLMERGED_merged['race'].astype("category")


formula_all = ("Q('Speed (GPS)') ~ Q('Stroke Rate') + Q('Effective Length') + "
               "Q('Max Force Angle') + PowerFit_Interaction3_z + SD_Power +"
               #"PowerFit_Interaction2 + PowerFit_Interaction3 +"
               "SD_Effective_Catch + SD_Effective_Finish + Avg_Power_per_Stroke"
              )

md_all = smf.mixedlm(formula_all, ALLMERGED_merged, groups=ALLMERGED_merged["race"])
mixed_model_all = md_all.fit()
print("=== Mixed Model: All Races ===")
print(mixed_model_all.summary())



four_rowers = (
    ALLMERGED_merged.groupby("race")["rower"]
    .nunique()
    .loc[lambda x: x == 4]
    .index
)



four = ALLMERGED_merged[ALLMERGED_merged['race'].isin(four_rowers)].copy()
four['race'] = four['race'].astype("category")

four = four.dropna(subset=[
    'Speed (GPS)', 'Stroke Rate', 'SD_Power', 'Effective Length',
    'Max Force Angle', 'PowerFit_Interaction1_z',
    'PowerFit_Interaction2_z', 'PowerFit_Interaction3_z', 'Avg_Power_per_Stroke',
    'SD_Effective_Catch', 'SD_Effective_Catch'
]).reset_index(drop=True)


md_four = smf.mixedlm(formula_all, four, groups=four["race"])
mixed_model_four = md_four.fit()
print("\n=== Mixed Model: Only Boats with 4 Rowers ===")
print(mixed_model_four.summary())


import statsmodels.formula.api as smf

md = smf.mixedlm(
    "Q('Speed (GPS)') ~ Q('Stroke Rate') + Power + Q('Effective Length') + Q('Max Force Angle') + Desync_Index + Wind_Along + Wind_Cross",
    data=ALLMERGED_var,
    groups="race",
    re_formula="~Q('Stroke Rate')"
)

mdf = md.fit()
print(mdf.summary())



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.lmplot(
    data=ALLMERGED_var, 
    x='Desync_Index', 
    y='Speed (GPS)', 
    hue='race', 
    col='race', 
    col_wrap=4, 
    height=3.5, 
    aspect=1,
    scatter_kws={'alpha': 0.4, 's': 20}, 
    line_kws={'linewidth': 2}
)

plt.subplots_adjust(top=0.92)
plt.suptitle("Effect of Desync Index on Speed by Race", fontsize=16)
plt.show()


plt.figure(figsize=(10, 6))
ax = sns.pointplot(
    data=ALLMERGED_var,
    x='race',
    y='Speed (GPS)',
    hue='Desync_Index',  
    dodge=True,
    ci='sd',
    palette='viridis'
)

ax.legend_.remove()  
plt.title("Mean Speed by Race")
plt.ylabel("Speed (GPS)")
plt.xlabel("Race")
plt.show()


#each race modeled

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

desync_results = []

for race_id in ALLMERGED_var['race'].unique():
    race_df = ALLMERGED_var[ALLMERGED_var['race'] == race_id].copy()

    X = race_df[['Desync_Index']]
    y = race_df['Speed (GPS)']


    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    desync_results.append({
        'race': race_id,
        'desync_coef': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'n_obs': len(race_df)
    })

desync_df = pd.DataFrame(desync_results)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=desync_df, x='race', y='desync_coef', palette='coolwarm')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Effect of Desync Index on Speed by Race")
plt.xlabel("Race ID")
plt.ylabel("Desync → Speed Coefficient")
plt.xticks(rotation=45)
plt.tight_layout()
for i, row in desync_df.iterrows():
    plt.text(i, row['desync_coef'], f"R²={row['r2']:.2f}", ha='center', va='bottom', fontsize=9)

plt.show()













