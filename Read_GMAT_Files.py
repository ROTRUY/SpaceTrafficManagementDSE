import pandas as pd

def Get_Eclipse_Data(fileName):
    eclipse_data = pd.read_csv(
        fileName,
        sep=r"\s{2,}",
        engine="python",
        skiprows=1,
        skipfooter=4,
        header=0
    )

    eclipse_data["Start Time (UTC)"] = pd.to_datetime(eclipse_data["Start Time (UTC)"], format="%d %b %Y %H:%M:%S.%f", utc=True)
    eclipse_data["Stop Time (UTC)"] = pd.to_datetime(eclipse_data["Stop Time (UTC)"], format="%d %b %Y %H:%M:%S.%f", utc=True)
    eclipse_data["Duration (s)"] = eclipse_data["Duration (s)"].astype(float)
    eclipse_data["Event Number"] = eclipse_data["Event Number"].astype(int)
    eclipse_data["Total Duration (s)"] = eclipse_data["Total Duration (s)"].astype(float)
    eclipse_data["Occ Body"] = eclipse_data["Occ Body"].astype("category")
    eclipse_data["Type"] = eclipse_data["Type"].astype("category")
    return eclipse_data

def Get_GroundPass_Data(fileName):
    groundpass_data = pd.read_csv(
        fileName,
        sep=r"\s{2,}",
        engine="python",
        skiprows=3,
        skipfooter=2,
        header=0
    )

    groundpass_data["Start Time (UTC)"] = pd.to_datetime(groundpass_data["Start Time (UTC)"], format="%d %b %Y %H:%M:%S.%f", utc=True)
    groundpass_data["Stop Time (UTC)"] = pd.to_datetime(groundpass_data["Stop Time (UTC)"], format="%d %b %Y %H:%M:%S.%f", utc=True)
    groundpass_data["Duration (s)"] = groundpass_data["Duration (s)"].astype(float)

    return groundpass_data