from pathlib import Path

import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------
# 1. FLIGHT SCHEDULE (YOUR INPUT)
# -----------------------------

FLIGHTS = [
    ("AI865","Air India","DEL","BOM",1137),
    ("6E2131","IndiGo","DEL","BOM",1137),
    ("QP1102","Akasa Air","DEL","BOM",1137),

    ("AI506","Air India","DEL","BLR",1740),
    ("6E2487","IndiGo","DEL","BLR",1740),
    ("UK811","Vistara","DEL","BLR",1740),

    ("6E2112","IndiGo","DEL","HYD",1253),
    ("AI542","Air India","DEL","HYD",1253),

    ("6E2024","IndiGo","DEL","MAA",1757),
    ("AI439","Air India","DEL","MAA",1757),

    ("6E2038","IndiGo","DEL","CCU",1305),
    ("AI701","Air India","DEL","CCU",1305),

    ("6E5021","IndiGo","DEL","AMD",775),
    ("AI817","Air India","DEL","AMD",775),

    ("6E2125","IndiGo","DEL","GOX",1515),

    ("6E5308","IndiGo","BOM","DEL",1137),

    ("AI677","Air India","BOM","BLR",842),
    ("6E5183","IndiGo","BOM","BLR",842),
    ("QP1362","Akasa Air","BOM","BLR",842),

    ("6E5384","IndiGo","BOM","HYD",617),
    ("AI617","Air India","BOM","HYD",617),

    ("6E278","IndiGo","BOM","MAA",1028),
    ("AI570","Air India","BOM","MAA",1028),

    ("6E335","IndiGo","BOM","CCU",1652),
    ("AI627","Air India","BOM","CCU",1652),

    ("6E5323","IndiGo","BOM","AMD",441),
    ("AI605","Air India","BOM","AMD",441),

    ("6E5214","IndiGo","BOM","GOX",435),
    ("QP1380","Akasa Air","BOM","GOX",435),

    ("6E2015","IndiGo","BLR","DEL",1740),
    ("AI505","Air India","BLR","DEL",1740),

    ("6E7164","IndiGo","BLR","HYD",500),
    ("AI514","Air India","BLR","HYD",500),

    ("6E214","IndiGo","BLR","MAA",270),
    ("6E6645","IndiGo","BLR","CCU",1560),
    ("AI772","Air India","BLR","CCU",1560),

    ("6E552","IndiGo","BLR","AMD",1235),
    ("6E6307","IndiGo","BLR","GOX",480),

    ("6E2516","IndiGo","HYD","DEL",1253),
    ("AI2626","Air India","HYD","BOM",617),

    ("6E946","IndiGo","HYD","BLR",500),
    ("6E654","IndiGo","HYD","MAA",520),
    ("AI560","Air India","HYD","MAA",520),

    ("6E7306","IndiGo","HYD","CCU",1180),
    ("6E6974","IndiGo","HYD","AMD",875),
    ("6E121","IndiGo","HYD","GOX",540),

    ("6E521","IndiGo","MAA","DEL",1757),
    ("6E277","IndiGo","MAA","BOM",1028),
    ("6E973","IndiGo","MAA","BLR",270),
    ("6E633","IndiGo","MAA","HYD",520),
    ("6E6144","IndiGo","MAA","CCU",1360),
    ("6E6344","IndiGo","MAA","AMD",1370),

    ("SG1077","SpiceJet","MAA","GOX",730),

    ("6E634","IndiGo","CCU","DEL",1305),
    ("AI628","Air India","CCU","BOM",1652),
    ("6E948","IndiGo","CCU","BLR",1560),
    ("6E7307","IndiGo","CCU","HYD",1180),

    ("QP1562","Akasa Air","CCU","GOX",1710)
]

flights_df = pd.DataFrame(
    FLIGHTS,
    columns=["flight_number","airline","origin","destination","distance_km"]
)

flights_df = pd.DataFrame(
    FLIGHTS,
    columns=["flight_number","airline","origin","destination","distance_km"]
)

# -----------------------------
# 2. SIMULATION SETTINGS
# -----------------------------

YEARS = 5
DAYS = 365 * YEARS
START_DATE = pd.Timestamp("2019-01-01")

rows = []

# -----------------------------
# 3. ROUTE RELIABILITY
# -----------------------------

route_effects = {
"DEL_BOM":4,
"DEL_BLR":6,
"DEL_HYD":3,
"BOM_BLR":2
}

# -----------------------------
# 4. AIRLINE RELIABILITY
# -----------------------------

airline_effects = {
"Air India":3,
"IndiGo":1,
"SpiceJet":4,
"Vistara":2,
"Akasa Air":2
}

# -----------------------------
# 5. WEATHER GENERATOR
# -----------------------------

def generate_weather(month):

    if month in [6,7,8,9]:   # monsoon
        precip = np.random.gamma(3,2)
        storm = np.random.binomial(1,0.25)
        visibility = np.random.normal(6,2)

    elif month in [12,1,2]:  # winter
        precip = np.random.gamma(1,1)
        storm = np.random.binomial(1,0.05)
        visibility = np.random.normal(8,2)

    else:                    # normal
        precip = np.random.gamma(1,0.5)
        storm = np.random.binomial(1,0.02)
        visibility = np.random.normal(10,2)

    wind = abs(np.random.normal(15,5))
    snow = np.random.binomial(1,0.01)
    temp = np.random.normal(28,6)

    return temp,precip,wind,visibility,storm,snow


# -----------------------------
# 6. DATA GENERATION LOOP
# -----------------------------

previous_delay = 0

for i in range(DAYS):

    date = START_DATE + pd.Timedelta(days=i)

    year = date.year
    month = date.month
    day_of_week = date.weekday()
    is_weekend = int(day_of_week >= 5)

    for _, flight in flights_df.iterrows():

        origin = flight.origin
        destination = flight.destination
        airline = flight.airline
        flight_number = flight.flight_number
        distance_km = flight.distance_km

        route = f"{origin}_{destination}"

        # -----------------------------
        # departure hour
        # -----------------------------

        dep_hour = np.random.randint(5,23)

        # -----------------------------
        # airport congestion
        # -----------------------------

        dep_flights_last_hr = np.random.poisson(12)
        arr_flights_last_hr = np.random.poisson(12)

        traffic = dep_flights_last_hr + arr_flights_last_hr

        # -----------------------------
        # weather
        # -----------------------------

        temp,precip,wind,visibility,storm,snow = generate_weather(month)

        # -----------------------------
        # delay components
        # -----------------------------

        route_effect = route_effects.get(route,3)

        airline_effect = airline_effects.get(airline,2)

        propagation_effect = previous_delay * 0.5

        traffic_effect = max(0,traffic-18) * 2

        weather_effect = (
            storm*np.random.uniform(8,15)
            + snow*np.random.uniform(10,18)
            + precip*np.random.uniform(0.5,1.5)
            + max(0,6-visibility)*2
        )

        base_delay = np.random.normal(4,5)

        noise = np.random.normal(0,3)

        # -----------------------------
        # final delay
        # -----------------------------

        delay_minutes = (
            base_delay
            + route_effect
            + airline_effect
            + propagation_effect
            + traffic_effect
            + weather_effect
            + noise
        )

        delay_minutes = max(0,delay_minutes * 0.5)

        delay_15 = int(delay_minutes > 15)

        previous_delay = delay_minutes

        rows.append([
            flight_number,
            airline,
            origin,
            destination,
            distance_km,
            year,
            month,
            day_of_week,
            is_weekend,
            dep_hour,
            temp,
            precip,
            wind,
            visibility,
            storm,
            snow,
            dep_flights_last_hr,
            arr_flights_last_hr,
            route,
            delay_minutes,
            delay_15
        ])


# -----------------------------
# 7. CREATE DATAFRAME
# -----------------------------

columns = [
"flight_number",
"airline",
"origin",
"destination",
"distance_km",
"year",
"month",
"day_of_week",
"is_weekend",
"dep_hour",
"temperature_c",
"precip_mm",
"wind_kmh",
"visibility_km",
"storm",
"snow",
"dep_flights_last_hr",
"arr_flights_last_hr",
"route",
"delay_minutes",
"delay_15"
]

df = pd.DataFrame(rows,columns=columns)

# -----------------------------
# 8. SAVE DATASET
# -----------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_DIR / "airline_delay_dataset.csv",index=False)

print("Dataset size:",df.shape)

print("Delay rate:")
print(df["delay_15"].value_counts(normalize=True))
