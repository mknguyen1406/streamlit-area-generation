import io

import streamlit as st
import pydeck as pdk

import numpy as np
import pandas as pd

import boto3
import requests
import json

import matplotlib.pyplot as plt
from geopy.distance import great_circle

from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

from settings import *


# get AWS S3 credentials
@st.cache
def get_aws_s3_credentials(url):

    r = requests.get(url)
    res = json.loads(r.text)

    access_id = res["access_id"]
    access_key = res["access_key"]
    region_name = res["region_name"]
    bucket_name = res["bucket_name"]

    return access_id, access_key, region_name, bucket_name

url = "https://prod-27.westus.logic.azure.com:443/workflows/e0fa0a5ff60f4bf9a418fe6a31585555/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=TDsIHvNfH-L-e7kCpQ5NfBgKQ_V-Ti-QuNPUQSYb3gc"

access_id, access_key, region_name, bucket_name = get_aws_s3_credentials(url)

###########################################################################################
#################################### Area Generation ######################################
###########################################################################################

# Add an app title
st.markdown('# Area Generation')

#####################################################################################
st.markdown('## 1. Load data')

@st.cache
def load_data(path):
    return pd.read_csv(path, sep=";")

# read file from S3
@st.cache
def read_file_from_s3(access_id, access_key, region_name, file):

    # Creating the low level functional client
    client = boto3.client(
        's3',
        aws_access_key_id = access_id,
        aws_secret_access_key = access_key,
        region_name = region_name
    )

    # Fetch the list of existing buckets
    clientResponse = client.list_buckets()

    # Print the bucket names one by one
    print('Printing bucket names...')
    for bucket in clientResponse['Buckets']:
        print(f'Bucket Name: {bucket["Name"]}')

    # get file
    response = client.get_object(Bucket=bucket_name, Key=file)

    # get status
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    # load data
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_csv(response.get("Body"), sep=";")
        print(df.head())
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    return df

# path = "../data/data_joined.csv"
# df = load_data(path)

df = read_file_from_s3(
    access_id=access_id, 
    access_key=access_key, 
    region_name=region_name, 
    file="kcom/data_joined.csv"
)

# create viz
st.write("First 5 rows for full data set in focus areas:")
st.write(df.head())
st.write("Number of postcodes: ", len(df))
st.write("Number of properties: ", df["Properties"].sum())

#####################################################################################
st.markdown('## 2. Apply filter')

st.markdown("""
Filters to be applied:
* NOT `In BT Fibre first`
* NOT `Urban major conurbation`
* Download speed availability NOT `>= 300 Mbit/s`
* `Properties` > 0
* NOT `FFE Area`
""")

# filter rows
@st.cache
def filter_rows(df, bt_fibre_first, not_urban, not_speed, min_property, ffe):
    df_filtered_rows = df[
        (df["BT Fibre First"] == bt_fibre_first) &
        (df["Rural or Urban"] != not_urban) &
        (df["Download speed availability"] != not_speed) &
        (df["Properties"] > min_property) &
        (df["FFE"] == ffe)
    ]
    return df_filtered_rows

df_filtered_rows = filter_rows(
    df=df,
    bt_fibre_first="Not in BT Plan",
    not_urban="Urban major conurbation",
    not_speed=">=300 Mbit/s",
    min_property=0,
    ffe=False,
)

# filter columns
@st.cache(allow_output_mutation=True)
def filter_columns(df_filtered_rows, cols):
    df_filtered_cols = df_filtered_rows[cols]
    return df_filtered_cols

cols = [
    "Properties",
    "Parish",
    "postcodex",
    "long",
    "lat"
]
df_filtered = filter_columns(df_filtered_rows, cols)

# create viz
st.write("First 5 rows for filtered data set in focus areas:")
st.write(df_filtered.head())
st.write("Number of postcodes: ", len(df_filtered))
st.write("Number of properties: ", df_filtered["Properties"].sum())

#####################################################################################
st.markdown('## 3. Perform clustering')

# extract coordinates
coords = df_filtered[['lat', 'long']].to_numpy()

# perform clustering
@st.cache
def perform_clustering(search_radius_km, kms_per_radian, min_samples):
    epsilon = search_radius_km / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))

    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

    return cluster_labels, num_clusters, clusters

# params

# add slider for search radius for clustering
st.write("Please set the search radius using the slider below. The clustering algorithm will search for and add points to clusters which are within this radius.")
search_radius_km = st.slider(
    label='Search Radius [M]',
    min_value=100, 
    max_value=5000, 
    value=500,
    step=100,
)
kms_per_radian = 6371.0088
min_samples=1

cluster_labels, num_clusters, clusters = perform_clustering(
    search_radius_km=search_radius_km/1000, 
    kms_per_radian=kms_per_radian,
    min_samples=min_samples,
)

# get cluster centers
@st.cache
def get_cluster_centers(df_filtered, clusters):

    # get centermost points
    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    centermost_points = clusters.map(get_centermost_point)


    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({'lon':lons, 'lat':lats})

    # create data frame
    return rep_points.apply(lambda row: df_filtered[(df_filtered['lat']==row['lat']) & (df_filtered['long']==row['lon'])].iloc[0], axis=1)

rs = get_cluster_centers(
    df_filtered=df_filtered,
    clusters=clusters
)

# create viz
st.write("First 10 rows for cluster centers:")
st.write(rs.head(10))
st.write("Number of clusters: ", num_clusters)

#####################################################################################
st.markdown('## 4. Generate hubs')

@st.cache
def get_hubs(df_filtered, cluster_labels, min_hub_size):

    # add cluster label to main data set
    df_filtered_with_cluster_labels = df_filtered.copy()
    df_filtered_with_cluster_labels["Cluster"] = cluster_labels

    # get number of properties per cluster
    df_grouped = df_filtered_with_cluster_labels.groupby("Cluster").sum()

    # join long/lat to cluster centers
    df_grouped = df_grouped[["Properties"]].join(rs[["long", "lat", "postcodex", "Parish"]])

    # filter hubs by minimum property size
    df_hub_clusters = df_grouped[df_grouped["Properties"] > min_hub_size]

    # filter spoke clusters by minimum property size
    df_spoke_clusters = df_grouped[df_grouped["Properties"] <= min_hub_size]

    print("No of hubs: ", len(df_hub_clusters))
    print("No of unique parishes: ", len(df_hub_clusters["Parish"].unique()))
    print("No of properties in hubs: ", df_hub_clusters["Properties"].sum())

    # join hub data to main data set
    df_hub_clusters["Hub"] = 1 # identifier for hub
    df_filtered_with_hubs = df_filtered_with_cluster_labels.merge(
        df_hub_clusters.reset_index()[["Cluster", "Hub"]],
        how="left",
        on="Cluster"
    )
    df_filtered_with_hubs["Hub"] = df_filtered_with_hubs["Hub"].fillna(0) # identifier for non-hubs

    # join number of properties 
    df_cluster_properties = df_grouped.reset_index()[["Cluster", "Properties"]]
    df_cluster_properties = df_cluster_properties.rename(columns={"Properties": "Properties_Cluster"})

    df_filtered_with_hubs = df_filtered_with_hubs.merge(
        df_cluster_properties,
        how="left",
        on="Cluster"
    )

    # define help column for reference
    df_hub_clusters["Hub_ID"] = list(range(len(df_hub_clusters)))

    return df_hub_clusters, df_filtered_with_hubs, df_spoke_clusters

# add slider for minimum size of hubs
st.write("Please set the minimum number of properties for a hub using the slider below. Clusters with enough properties become hubs.")
min_hub_size = st.slider(
    label='Minimum number of properties per hub',
    min_value=500, 
    max_value=5000, 
    value=1000,
    step=500,
)

df_hub_clusters, df_filtered_with_hubs, df_spoke_clusters = get_hubs(
    df_filtered=df_filtered,
    cluster_labels=cluster_labels,
    min_hub_size=min_hub_size
)

# create viz
st.write("Hub clusters:")
st.write(df_hub_clusters.reset_index())

st.write("Number of hubs after filtering: ", len(df_hub_clusters))
st.write("Number of properties in hubs: ", df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 1]["Properties"].sum())
st.write("Number of postcodes in hubs: ", len(df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 1]))
st.write("Number of properties not in hubs: ", df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 0]["Properties"].sum())
st.write("Number of postcodes not in hubs: ", len(df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 0]))

#####################################################################################
st.markdown('## 5. Add spokes')

@st.cache
def generate_spokes(df_hub_clusters, df_filtered_with_hubs, kms_per_radian, threshold_radius, threshold_min_prop_km):

    # get hub coordinates
    hub_coords = df_hub_clusters[["lat", "long"]].to_numpy()

    # get spoke coordinates
    df_spokes = df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 0]
    spoke_coords = df_spokes[["lat", "long"]].to_numpy()

    # calculate distance matrix between spokes and hubs
    dist_matrix = haversine_distances(X=np.radians(spoke_coords), Y=np.radians(hub_coords))
    dist_matrix = dist_matrix * kms_per_radian  # multiply by Earth radius to get kilometers

    spoke_cluster = []
    for i in range(len(dist_matrix)):
    
        # get distance from postcode to nearest hub
        spoke_dist = dist_matrix[i]
        nearest_hub_id = spoke_dist.argmin()
        distance = spoke_dist[nearest_hub_id]
        
        # get number of properties in cluster per kilometer
        properties_in_cluster = df_spokes.iloc[i,:]["Properties_Cluster"]
        properties_in_cluster_per_km = properties_in_cluster / distance
        
        # if threshold met, get hub cluster - otherwise assign -1 as cluster
        if (distance < threshold_radius) & (properties_in_cluster_per_km > threshold_min_prop_km):
            cluster = df_hub_clusters[df_hub_clusters["Hub_ID"] == nearest_hub_id].index[0]
            spoke_cluster.append(cluster)
        else:
            spoke_cluster.append(-1)

    df_spokes["Cluster"] = spoke_cluster

    return df_spokes

# add slider for search radius for clustering
st.write("Please select a threshold radius within which to search for postcodes to be added as spokes to the hubs. All remaining postcodes are outliers which won't be considered in the area selection.")
threshold_radius = st.slider(
    label='Threshold radius [KM]',
    min_value=0, 
    max_value=20, 
    value=5,
    step=1,
)
st.write("Please select a threshold for the minumum number of properties per kilometer for clusters to be added as spokes to the hubs. All remaining postcodes are outliers which won't be considered in the area selection.")
threshold_min_prop_km = st.slider(
    label='Minimum properties per KM as Backhaul',
    min_value=0, 
    max_value=100, 
    value=2,
    step=1,
)
df_spokes = generate_spokes(
    df_hub_clusters=df_hub_clusters,
    df_filtered_with_hubs=df_filtered_with_hubs,
    kms_per_radian=kms_per_radian,
    threshold_radius=threshold_radius,
    threshold_min_prop_km=threshold_min_prop_km,
)

# create viz
st.write("First 100 spoke clusters:")
st.write(df_spoke_clusters.head(100))

st.write("Number of postcodes without a cluster: ", len(df_spokes[df_spokes["Cluster"] == -1]))
st.write("Number of properties without a cluster: ", df_spokes[df_spokes["Cluster"] == -1]["Properties"].sum())
st.write("Number of postcodes with a cluster (hubs & spokes): ", len(df_spokes[df_spokes["Cluster"] != -1]) + len(df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 1]))
st.write("Number of properties with a cluster (hubs & spokes): ", df_spokes[df_spokes["Cluster"] != -1]["Properties"].sum() + df_filtered_with_hubs[df_filtered_with_hubs["Hub"] == 1]["Properties"].sum())

#####################################################################################
st.markdown('## 6. Area Cluster Overview')

# generate data frame that contains all hub and spoke information
@st.cache
def generate_final_area_df(df_filtered_with_hubs, df_spokes):
    df_hub_spokes = df_filtered_with_hubs.join(
        df_spokes[["Cluster"]], 
        how="left", 
        lsuffix="_Individual",
        rsuffix="_Hub",
    )

    # Cluster_Hub to contain assigned Hub - Cluster_Individual to contain individual cluster for every postcode
    df_hub_spokes["Cluster_Hub"] = df_hub_spokes["Cluster_Hub"].fillna(df_hub_spokes["Cluster_Individual"])

    # get total number of properties per cluster in hubs and spokes
    df_areas_final = df_hub_spokes.groupby("Cluster_Hub").sum()[["Properties"]]
    df_areas_final.columns = ["Properties_Total"]
    
    # join number of properties in hub only and calculate the ones in spokes only
    hub_col = ["Properties", "long", "lat", "postcodex", "Parish"]
    df_areas_final = df_areas_final.join(
        df_hub_clusters[hub_col],
        how="left"
    )
    df_areas_final = df_areas_final.rename(columns={"Properties": "Properties_Hub"})
    df_areas_final["Properties_Spoke"] = df_areas_final["Properties_Total"] - df_areas_final["Properties_Hub"]

    # change order of columns
    hub_col_reordered = ["Parish", "postcodex", "long", "lat", "Properties_Total", "Properties_Hub", "Properties_Spoke"]
    df_areas_final = df_areas_final[hub_col_reordered]

    return df_areas_final, df_hub_spokes

df_areas_final, df_hub_spokes = generate_final_area_df(df_filtered_with_hubs, df_spokes)


# create viz
st.write("Overview of number of properties in each area cluster by hub and spoke. First column depicts cluster ID where -1 indicates to postcodes without cluster")
st.write(df_areas_final)

#####################################################################################
st.markdown('## 7. Visualize clusters')

point_radius = 250
hub_radius = 500

# Plot map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=53.767750,
        longitude=-0.335827,
        zoom=5,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df_hub_spokes[df_hub_spokes["Cluster_Hub"] == -1][["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["mid_light_gray"]["rgba"],
            get_radius=point_radius,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_hub_spokes[(df_hub_spokes["Cluster_Hub"] != -1) & (df_hub_spokes["Hub"] == 0)][["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["mid_dark_purple"]["rgba"],
            get_radius=point_radius,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_hub_spokes[df_hub_spokes["Hub"] == 1][["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["dark_purple"]["rgba"],
            get_radius=point_radius,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_hub_clusters[["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["kcom"]["rgba"],
            get_radius=hub_radius,
        ),
    ],
))

#####################################################################################
st.markdown('## 8. Save clusters')
st.write("Use the button below to save the generated clusters used as in input for the area selection. All postcodes without cluster will be excluded.")

# write data to S3 bucket
def write_file_to_s3(access_id, access_key, region_name, bucket_name, df, file):

    # Creating the low level functional client
    client = boto3.client(
        's3',
        aws_access_key_id = access_id,
        aws_secret_access_key = access_key,
        region_name = region_name
    )

    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, index=False, sep=";")

        response = client.put_object(
            Bucket=bucket_name, Key=file, Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

@st.cache
def save_output(file, df_hub_spokes, df_filtered_rows):
    cluster_cols = [
        "Cluster_Individual",
        "Hub",
        "Cluster_Hub",
    ]

    # join raw data with generated clusters
    df_area_cluster_output = df_hub_spokes[cluster_cols].join(
        df_filtered_rows.reset_index(drop=True),
        how="left"
    )

    # exclude postcodes without a hub cluser
    df_area_cluster_output = df_area_cluster_output[df_area_cluster_output["Cluster_Hub"] != -1]

    # save to csv
    # df_area_cluster_output.to_csv(output_path, index=False)
    write_file_to_s3(
        access_id=access_id,
        access_key=access_key,
        region_name=region_name,
        bucket_name=bucket_name,
        df=df_area_cluster_output,
        file=file
    )

    return df_area_cluster_output

if st.button('Save'):

    # output_path = "../data/output/data_area_cluster.csv"
    output_file = "kcom/data_area_cluster.csv"
    df_area_cluster_output = save_output(
        file=output_file, 
        df_hub_spokes=df_hub_spokes, 
        df_filtered_rows=df_filtered_rows
    )
    print(f"Output data saved to: {output_file}")

    # create viz
    st.write("Number of postcodes in output dataset: ", len(df_area_cluster_output))
    st.write("Number of properties in output dataset: ", df_area_cluster_output["Properties"].sum())