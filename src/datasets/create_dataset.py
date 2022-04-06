import pandas as pds
import modin.pandas as pd
import numpy as np
import modin.config as cfg
cfg.Engine.put('ray')
cfg.Backend.put('pandas')

#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   the_geom    24715 non-null  object
#  1   marker_id   5933 non-null   object
#  2   meter_id    543 non-null    object
#  3   bay_id      24715 non-null  int64
#  4   last_edit   24624 non-null  float64
#  5   rd_seg_id   23571 non-null  float64
#  6   rd_seg_dsc  23571 non-null  object
# dtypes: float64(2), int64(1), object(4)

dtypes_location = {
    "the_geom" : str,
    "marker_id" : str,
    "meter_id" : str,
    "bay_id": int,
    "last_edit" : "float64",
    "rd_seg_id": "float64",
    "rd_seg_dsc": "object",

}

locations = pd.read_csv("../../data/bay_locations.csv", dtype=dtypes_location)


locations.info(verbose=True)


#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   DeviceId          500000 non-null  int64
#  1   ArrivalTime       500000 non-null  datetime64[ns]
#  2   DepartureTime     500000 non-null  datetime64[ns]
#  3   DurationMinutes   500000 non-null  int64
#  4   StreetMarker      500000 non-null  object
#  5   SignPlateID       356181 non-null  float64
#  6   Sign              356181 non-null  object
#  7   AreaName          499999 non-null  object
#  8   StreetId          500000 non-null  int64
#  9   StreetName        500000 non-null  object
#  10  BetweenStreet1ID  500000 non-null  int64
#  11  BetweenStreet1    500000 non-null  object
#  12  BetweenStreet2ID  500000 non-null  int64
#  13  BetweenStreet2    500000 non-null  object
#  14  SideOfStreet      500000 non-null  int64
#  15  SideOfStreetCode  500000 non-null  object
#  16  SideName          500000 non-null  object
#  17  BayId             500000 non-null  int64
#  18  InViolation       500000 non-null  bool
#  19  VehiclePresent    500000 non-null  bool
# dtypes: bool(2), datetime64[ns](2), float64(1), int64(8), object(8)


#columns needed: DeviceId, ArrivalTime, DepartureTime, Sign, VehiclePresent, StreetMarker, AreaName

dtypes = {
    "DeviceId" : str,
   # "ArrivalTime" : "datetime64[ns]",
 #   "DepartureTime": "datetime64[ns]",
    "Sign" : str,
    "VehiclePresent" : bool,
    "StreetMarker": str,
    "AreaName" : str
}

usecols = list(dtypes.keys()) + ["ArrivalTime", "DepartureTime"]

parking_events = pd.read_csv("../../data/On-street_Car_Parking_Sensor_Data_-_2019.csv",
                             infer_datetime_format=True,
                             engine="c",
                             parse_dates=['ArrivalTime', 'DepartureTime'],
                             dtype=dtypes,
                             usecols=usecols)

parking_events.info(verbose=True)

#remove events without area name
parking_events = parking_events.dropna(subset=["AreaName"])

#remve events without vehicle present
parking_events = parking_events[parking_events["VehiclePresent"]]

#calculate duration of event
parking_events["DurationSeconds"] = (parking_events["DepartureTime"] - parking_events["ArrivalTime"]).dt.total_seconds().astype(int)

print(len(parking_events["DurationSeconds"][parking_events["DurationSeconds"]<0]))

#remove negative durations
parking_events = parking_events[parking_events["DurationSeconds"]>=0]






parking_events["Sign"] = parking_events["Sign"].replace({np.nan: "None"})

parking_events["MaxMinutes"] = np.select(
    [
        parking_events["Sign"].str.startswith("None"),
        parking_events["Sign"].str.startswith("P/5"),
        parking_events["Sign"].str.startswith("P5"),
        parking_events["Sign"].str.startswith("P10"),
        parking_events["Sign"].str.startswith("P/10"),
        parking_events["Sign"].str.startswith("P/15"),
        parking_events["Sign"].str.startswith("1/4P"),
        parking_events["Sign"].str.startswith("1/4 P"),
        parking_events["Sign"].str.startswith("1/2P"),
        parking_events["Sign"].str.startswith("1/2"),
        parking_events["Sign"].str.startswith("1P"),
        parking_events["Sign"].str.startswith("2P"),
        parking_events["Sign"].str.startswith("3P"),
        parking_events["Sign"].str.startswith("4P"),
        parking_events["Sign"].str.startswith("10P"),
        parking_events["Sign"].str.startswith("LZ 15M"),
        parking_events["Sign"].str.startswith("LZ 30M"),
        parking_events["Sign"].str.contains("60mins"),
        parking_events["Sign"].str.contains("30MINS"),
        parking_events["Sign"].str.contains("15mins"),
        parking_events["Sign"].str.contains("15Mins"),
        parking_events["Sign"].str.contains("1PMTR"),
    ],
    [
        "None",
        5,
        5,
        10,
        10,
        15,
        15,
        15,
        30,
        30,
        60,
        120,
        180,
        240,
        240,
        15,
        30,
        60,
        30,
        15,
        15,
        60
    ],
    default="Unknown"
)

# Drop rows with unknown sign? use 60 minutes instead? Or use Duration of the event?
parking_events = parking_events[parking_events["MaxMinutes"] != "Unknown"]
parking_events = parking_events[parking_events["MaxMinutes"] != "None"]
parking_events['MaxMinutes'] = parking_events['MaxMinutes'].astype(float)
parking_events["MaxSeconds"] = (parking_events["MaxMinutes"] * 60).astype(int)

parking_events["Overstayed"] = parking_events["MaxSeconds"]< parking_events["DurationSeconds"]
parking_events["OverstayDuration"] = 0
parking_events.loc[parking_events["Overstayed"],"OverstayDuration"] = parking_events["DurationSeconds"] - parking_events["MaxSeconds"]

events_with_location = pd.merge(parking_events, locations, left_on="StreetMarker", right_on="marker_id")


arrivals = events_with_location[["ArrivalTime", "StreetMarker", "MaxSeconds", "AreaName"]].copy()
arrivals.rename({"ArrivalTime": "Time"}, axis=1, inplace=True)
arrivals["Type"] = "Arrival"


#%%

departures = events_with_location[["DepartureTime", "StreetMarker", "MaxSeconds", "AreaName"]].copy()
departures.rename({"DepartureTime": "Time"}, axis=1, inplace=True)
departures["Type"] = "Departure"
#todo maybe set maxseconds to 0 for depatures (or not sometimes this makes sense)

#%%

violations = events_with_location[events_with_location["Overstayed"]][["ArrivalTime", "StreetMarker", "MaxSeconds", "AreaName"]].copy()
violations["Time"] = pds.to_datetime(violations["ArrivalTime"]) + pds.to_timedelta(violations["MaxSeconds"], unit='s')
    #violations.apply(lambda x: x["ArrivalTime"] + pd.DateOffset(seconds=x["MaxSeconds"]), axis=1)
violations.drop(['ArrivalTime'], axis=1, inplace=True)
violations["Type"] = "Violation"


#%%

event_log = arrivals.append(departures).append(violations)
event_log.sort_values(by=["Time"], inplace=True)
event_log.reset_index(drop=True, inplace=True)

#%%

event_log.to_pickle('../../data/event_log_2019.gzip',compression='gzip')
