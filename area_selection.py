import streamlit as st
import altair as alt
import pydeck as pdk

import numpy as np
import pandas as pd

import boto3
import requests
import json

from sklearn.impute import SimpleImputer

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
#################################### Area Selection #######################################
###########################################################################################

# Add an app title
st.markdown('# Area Selection')

#####################################################################################
st.markdown('## 1. Load data')

@st.cache
def load_data(path):
    return pd.read_csv(path, sep=";")

# read file from S3
@st.cache
def read_file_from_s3(access_id, access_key, region_name, bucket_name, file):

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

# path = "../data/output/data_area_cluster.csv"
# df = load_data(path)

df = read_file_from_s3(
    access_id=access_id, 
    access_key=access_key, 
    bucket_name=bucket_name,
    region_name=region_name, 
    file="kcom/data_area_cluster.csv"
)

# create viz
st.write("First 5 postcodes for full data set in focus areas:")
st.write(df.head())
st.write("Number of postcodes: ", len(df))
st.write("Number of properties: ", df["Properties"].sum())

#####################################################################################
st.markdown('## 2. Generate demand score')
st.markdown('### 2.1 Select columns')

st.markdown("""
Columns to be selected:
* `kcom_% of premises unable to receive 30Mbit/s`
* `demo_res_% Persons aged 16 and over whose highest level of qualification is Level 4 and above`
* `imd_Crime Rank (where 1 is most deprived)`
""")

# filter columns
@st.cache
def filter_cols(df, cols):
    return df[cols]

select_cols = [
    "kcom_% of premises unable to receive 30Mbit/s",
    "demo_res_% Persons aged 16 and over whose highest level of qualification is Level 4 and above",
    "imd_Crime Rank (where 1 is most deprived)",
]

df_filtered = filter_cols(df, select_cols)

# create viz
st.write("First 5 postcodes of data set with selected columns:")
st.write(df_filtered.head())


#####################################################################################
st.markdown('### 2.2 Impute missing values')

st.write("Missing values will be replaced with mean of column.")

# filter columns
@st.cache
def impute_df(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    return df_imputed

df_imputed = impute_df(df_filtered)

# create viz
st.write("First 5 postcodes of data set with imputed values:")
st.write(df_imputed.head())


#####################################################################################
st.markdown('### 2.3 Scale values')

st.write("All columns are scaled to a range of [0,1].")

# scale columns to range between 0 and 1
@st.cache
def scale_df(df):

    df_scaled = df.copy()

    # scale each column
    for col in df_scaled.columns:
        df_scaled[col] = (df_scaled[col] - df_scaled[col].min()) / (df_scaled[col].max() - df_scaled[col].min())

    return df_scaled

df_scaled = scale_df(df_imputed)

# create viz
st.write("First 5 postcodes of data set with scaled values:")
st.write(df_scaled.head())


#####################################################################################
st.markdown('### 2.4 Run regression on postcode level')

regression_coefficients = {
    "intercept": 0.0466607152438894,
    "kcom_% of premises unable to receive 30Mbit/s": 0.120019955783519,
    "demo_res_% Persons aged 16 and over whose highest level of qualification is Level 4 and above": 0.0402844845123129,
    "imd_Crime Rank (where 1 is most deprived)": 0.0657097811630079,
}

st.write("Coefficients for linear regression:")
st.write(pd.DataFrame(regression_coefficients, columns=["Variable", "Coefficient"]))

# run regression
@st.cache
def run_regression(df):

    # create deep copy
    df_demand_score = df.copy()

    df_demand_score["Demand_Score"] = \
        regression_coefficients["intercept"] + \
        regression_coefficients[select_cols[0]] * df_demand_score[select_cols[0]] + \
        regression_coefficients[select_cols[1]] * df_demand_score[select_cols[1]] + \
        regression_coefficients[select_cols[2]] * df_demand_score[select_cols[2]]

    return df_demand_score

df_demand_score = run_regression(df_scaled)

# create viz
st.write("First 5 postcodes of dataset with demand score:")
st.write(df_demand_score.head())


#####################################################################################
st.markdown('### 2.5 Aggregate postcodes into clusters')

st.write("Aggregate demand scores on postcode level into area clusters. Calculate weighted mean by number of properties.")

# run regression
@st.cache
def aggregate_to_clusters_demand(df_demand_score, df_full):

    # Add hub cluster and number of properties to postcodes
    df_tmp = df_demand_score.copy()
    df_tmp["Cluster_Hub"] = df_full["Cluster_Hub"]
    df_tmp["Properties"] = df_full["Properties"]

    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df.loc[x.index, "Properties"])
    
    # Groupby and aggregate with namedAgg [1]:
    df_demand_score_clusters = df_tmp.groupby(["Cluster_Hub"]).agg(Properties_Sum=("Properties", "sum"), Demand_Score_Avg=("Demand_Score", wm))

    # Assign demand rank
    df_demand_score_clusters["Demand_Rank"] = df_demand_score_clusters["Demand_Score_Avg"].rank(ascending=False)

    return df_demand_score_clusters

df_demand_score_clusters = aggregate_to_clusters_demand(df_demand_score, df)

# create viz
st.write("All clusters with weighted averaged demand score and rank (1 is best):")
st.write(df_demand_score_clusters)


#####################################################################################
st.markdown('## 3. Generate cost score')
st.markdown('### 3.1 Average distance between homes on postcode level')

# run regression
@st.cache
def calculate_avg_distance(df):

    # Select columns
    dens_cols = [
        "lsoa11",
        "dens_All usual residents",
        "dens_Area (Hectares)",
        "Properties",
        "Cluster_Hub"
    ]
    df_dens = df[dens_cols]

    # get LSOAs
    df_dens_lsoa = df_dens.groupby(["lsoa11"]).agg(Residents_Sum=("dens_All usual residents", "mean"), Area_Hectares=("dens_Area (Hectares)", "mean"))

    # calculate number of properies per LSOA - assume 56/23 residents per property
    df_dens_lsoa["Properties_Sum"] = df_dens_lsoa["Residents_Sum"] * 23/56

    # Calculate area in qm
    df_dens_lsoa["Area_qm"] = df_dens_lsoa["Area_Hectares"] * 10000

    # Calculate density for LSOA
    df_dens_lsoa["Properties_per_qm"] = df_dens_lsoa["Properties_Sum"] / df_dens_lsoa["Area_qm"]

    # Calculate avg distance between homes for LSOA
    df_dens_lsoa["Avg_distance_m"] = np.sqrt(1 / df_dens_lsoa["Properties_per_qm"])

    # join avg distance to postcode
    df_dens = df_dens.merge(
        df_dens_lsoa.reset_index()[["lsoa11", "Avg_distance_m"]],
        how="left",
        on="lsoa11"
    )

    return df_dens

df_dens = calculate_avg_distance(df)

# create viz
st.write("First 5 postcodes of dataset with avg. distance between homes in meter:")
st.write(df_dens.head())

#####################################################################################
st.markdown('### 3.2 Aggregate postcodes into clusters')

st.write("Aggregate average distance between homes on postcode level into area clusters. Calculate weighted mean by number of properties.")

# aggregate 
@st.cache
def aggregate_to_clusters_cost(df_dens):

    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df.loc[x.index, "Properties"])
    
    # Groupby and aggregate with namedAgg [1]:
    df_cost_score_clusters = df_dens.groupby(["Cluster_Hub"]).agg(Properties_Sum=("Properties", "sum"), Avg_distance_m=("Avg_distance_m", wm))

    # Assign cost rank
    df_cost_score_clusters["Cost_Rank"] = df_cost_score_clusters["Avg_distance_m"].rank(ascending=True)

    return df_cost_score_clusters

df_cost_score_clusters = aggregate_to_clusters_cost(df_dens)

# create viz
st.write("All clusters with weighted averaged distance between homes and rank (1 is best):")
st.write(df_cost_score_clusters)


#####################################################################################
st.markdown('## 4. Select areas')

# join final dataset
@st.cache
def join_clusters(df_cost_score_clusters, df_demand_score_clusters):
    
    # join clusters
    df_cluster_join = df_cost_score_clusters.join(
        df_demand_score_clusters[["Demand_Score_Avg", "Demand_Rank"]],
        how="left"
    )

    return df_cluster_join, len(df_cluster_join)

df_cluster_join, cluster_num = join_clusters(
    df_cost_score_clusters=df_cost_score_clusters,
    df_demand_score_clusters=df_demand_score_clusters
)

# make selection
@st.cache
def select_clusters(df_cluster_join, demand_rank_range, cost_rank_range, df):
    
    # slice clusters
    df_cluster_select = df_cluster_join[
        (df_cluster_join["Demand_Rank"] >= demand_rank_range[0]) &
        (df_cluster_join["Demand_Rank"] <= demand_rank_range[1]) &
        (df_cluster_join["Cost_Rank"] >= cost_rank_range[0]) &
        (df_cluster_join["Cost_Rank"] <= cost_rank_range[1]) 
    ]

    # selected cluster IDs
    select_cluster_list = list(df_cluster_select.reset_index()["Cluster_Hub"])

    # create hard copy of postcode dataset
    df_postcodes_select = df[df["Cluster_Hub"].isin(select_cluster_list)]
    df_postcodes_select_hubs = df_postcodes_select[df_postcodes_select["Hub"] == 1]
    df_postcodes_select_spokes = df_postcodes_select[df_postcodes_select["Hub"] == 0]

    return df_cluster_select, df_postcodes_select_hubs, df_postcodes_select_spokes

# create range sliders for rank selection
st.write("Please select areas to be included based on their demand and cost rank using the sliders below.")
demand_rank_range = st.slider(
    label='Demand Rank',
    min_value=1, 
    max_value=cluster_num, 
    value=[1, cluster_num],
    step=1,
)

cost_rank_range = st.slider(
    label='Cost Rank',
    min_value=1, 
    max_value=cluster_num, 
    value=[1, cluster_num],
    step=1,
)

df_cluster_select, df_postcodes_select_hubs, df_postcodes_select_spokes = select_clusters(
    df_cluster_join=df_cluster_join,
    demand_rank_range=demand_rank_range,
    cost_rank_range=cost_rank_range,
    df=df
)

# Create viz for stats
st.write("Number of selected area clusters: ", len(df_cluster_select))
st.write("Number of selected properties: ", df_cluster_select["Properties_Sum"].sum())

# create scatter plot
c = alt.Chart(df_cluster_select).mark_circle().encode(
    x='Cost_Rank', 
    y='Demand_Rank', 
    size='Properties_Sum', 
    # color='c', 
    tooltip=["Properties_Sum", 'Cost_Rank', 'Avg_distance_m', 'Demand_Rank', "Demand_Score_Avg"]
).configure_mark(
    opacity=0.5,
    color=colours["dark_purple"]["hex"]
).properties(
    width=500,
    height=500
).configure_axis(
    grid=False
)
st.altair_chart(c, use_container_width=True)

# Plot map
point_radius = 250
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
            data=df_postcodes_select_spokes[["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["mid_dark_purple"]["rgba"],
            get_radius=point_radius,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_postcodes_select_hubs[["long", "lat"]],
            get_position='[long, lat]',
            get_color=colours["dark_purple"]["rgba"],
            get_radius=point_radius,
        ),
    ],
))