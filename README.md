# Airbnb_Amsterdam_listings_proj_paandas
DUPLICATE THIS COLAB DOCUMENT TO START WORKING ON IT: On the top-left corner of this page, go to File > Save a copy to drive.
SHARE SETTINGS: In the new notebook, set the sharing settings to "Anyone with the link" by clicking "Share" on the top-right corner.


Week 2: Discover the Airbnb Dataset (and Filter It!)
Hi! 👋👋👋 Are you excited to start the second week's project for Python for Data Science?

This week's lecture and material on CoRise showed you how to clean a DataFrame and merge one DataFrame into another. As you might have noticed, our dataset this week has way more columns than our dataset from Week 1. This is much more true to real life. It's messy, it's bloated, every time is unique, and above all, it's MUCH MORE interesting!

For this project, we saved an older version of the "clean" Listings DataFrame so that you can apply all the steps we performed on the Calendar DataFrame directly to this older version and reproduce the methods you learned on slightly different data. Let's get started 💪💪!

Downloading the Dataset
You'll need to download some prerequisite Python packages in order to run all the code below. Let's install them!

[61]
2s
%%capture
!pip install numpy pandas streamlit gdown pyarrow
We will download the datasets from Google Drive just like we did last week, but this time the datasets are in Pickle and Parquet format.

[62]
1s
import os
import shutil

import gdown
import numpy as np
import pandas as pd

# Download files from Google Drive
# Based on data from: http://insideairbnb.com/get-the-data/
file_id_1 = "1m185vTdh-u7_A2ZElBvUD4SCO6oETll2"
file_id_2 = "1w41V1oWHJrBdaNJJQ4oxVBuml5CO7MQX"
downloaded_file_1 = "listings_project.pkl"
downloaded_file_2 = "calendar_project.parquet"
# Download the files from Google Drive
gdown.download(id=file_id_1, output=downloaded_file_1)
gdown.download(id=file_id_2, output=downloaded_file_2)

[63]
0s
# Show all columns (instead of cascading columns in the middle)
pd.set_option("display.max_columns", None)
# Don't show numbers in scientific notation
pd.set_option("display.float_format", "{:.2f}".format)
Preprocessing the Dataset
Please load the downloaded files as DataFrames (dfs). The method for loading these datasets is the same as what we did on the CoRise platform.

Task 1: Read Pickle and Parquet
[Related section on CoRise]

Read the Python Pickle and PyArrow Parquet files we've just downloaded as df_list and df_cal.

[64]
0s
from pandas.compat import pyarrow
df_list = pd.read_pickle("listings_project.pkl")
df_cal = pd.read_parquet("calendar_project.parquet", engine="pyarrow")


Now instead of cleaning the Calendar DataFrame, you are going to clean the Listings DataFrame. You will use the same steps we used to clean the Calendar data on the CoRise platform this week. Let's first get an overview of the columns that are in this particular DataFrame 🧐.

Task 2: Print column names, types, and non-null values
[Related section on CoRise]

Let's try and get an overview of the Listings DataFrame, called df_list. This should show us some details about the columns in the DataFrame, like the column names, their data types, and the number of non-null values.

[65]
0s
df_list.head()

[66]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6165 entries, 0 to 6172
Data columns (total 34 columns):
 #   Column                                Non-Null Count  Dtype  
---  ------                                --------------  -----  
 0   id                                    6165 non-null   int64  
 1   host_acceptance_rate                  5365 non-null   float64
 2   host_is_superhost                     6165 non-null   object 
 3   host_listings_count                   6165 non-null   int64  
 4   host_total_listings_count             6165 non-null   int64  
 5   neighbourhood_cleansed                6165 non-null   object 
 6   latitude                              6165 non-null   float64
 7   longitude                             6165 non-null   float64
 8   room_type                             6165 non-null   object 
 9   accommodates                          6165 non-null   int64  
 10  bedrooms                              5859 non-null   float64
 11  beds                                  6082 non-null   float64
 12  amenities                             6165 non-null   int64  
 13  price                                 6165 non-null   object 
 14  minimum_nights                        6165 non-null   int64  
 15  maximum_nights                        6165 non-null   int64  
 16  has_availability                      6165 non-null   object 
 17  availability_30                       6165 non-null   int64  
 18  availability_60                       6165 non-null   int64  
 19  availability_90                       6165 non-null   int64  
 20  availability_365                      6165 non-null   int64  
 21  number_of_reviews                     6165 non-null   int64  
 22  number_of_reviews_ltm                 6165 non-null   int64  
 23  number_of_reviews_l30d                6165 non-null   int64  
 24  review_scores_rating                  5581 non-null   float64
 25  instant_bookable                      6165 non-null   object 
 26  reviews_per_month                     5581 non-null   float64
 27  price_in_euros                        0 non-null      object 
 28  price_per_person                      6165 non-null   object 
 29  minimum_price                         6165 non-null   object 
 30  discount_per_5_days_booked            6165 non-null   object 
 31  discount_per_10_days_booked           6165 non-null   object 
 32  discount_per_30_and_more_days_booked  6165 non-null   object 
 33  service_cost                          6165 non-null   object 
dtypes: float64(7), int64(14), object(13)
memory usage: 1.6+ MB
[67]
0s
df_list.describe()

[68]
0s
df_list.shape
(6165, 34)
This info printout provides a good overview of which columns we need to investigate further. We saw on CoRise this week that columns with the dtype object and sometimes float require inspection and cleaning.

Let's start first with the discount_per_... columns, where your output should look somthing like this

0    5%
1    5%
2    7%
3    6%
4    9%
Name: discount_per_5_days_booked, dtype: object
[69]
0s
df_list.discount_per_5_days_booked.head(5)
0    5%
1    5%
2    7%
3    6%
4    9%
Name: discount_per_5_days_booked, dtype: object
Task 3: Remove, convert, and format
[Related section on CoRise]

Perform this four-step process to change each of the three discount_per_... columns into their proper format:

Remove non-numeric characters, like the percent symbol, so you can perform mathematical calculations on the column
Change the column into a float data type in order to convert the data into a ratio
Multiply the whole column by 0.01 so you end up with a probability ratio instead of a percentage
Overwrite the old discount_per_... column with this new column
Perform these four steps for all thee columns.

Please note that running this code block more than once might cause an error. This is because you are re-assigning your columns with this code, and if you run the code again, the variable/column you are referring to has already been changed to its preferred state.

[70]
0s
df_list["discount_per_5_days_booked"] = df_list["discount_per_5_days_booked"].str.replace("%", " ", regex=True).astype("float") * 0.01
df_list["discount_per_10_days_booked"] = df_list["discount_per_10_days_booked"].str.replace("%", " ", regex=True).astype("float") * 0.01
df_list["discount_per_30_and_more_days_booked"] = df_list["discount_per_30_and_more_days_booked"].str.replace("%", " ", regex=True).astype("float") * 0.01
Awesome! Let's inspect our results. Your column output should look something like this:

0   0.05
1   0.05
2   0.07
3   0.06
4   0.09
Name: discount_per_5_days_booked, dtype: float64
[71]
0s
df_list.discount_per_5_days_booked.head(5)
0   0.05
1   0.05
2   0.07
3   0.06
4   0.09
Name: discount_per_5_days_booked, dtype: float64
This data looks great for performing calculations!

Next, the columns host_is_superhost, instant_bookable, and has_availability are all boolean columns in the sense that their data represents true and false values, but currently are recognized as objects.

[72]
0s
df_list[["host_is_superhost", "instant_bookable", "has_availability"]].head(5)

This is because the letters in these columns (t and f) are written as strings and not as boolean data types. This means we need to replace our string values with the boolean equivalent dtype.

Task 4: Booleans!
[Related section on CoRise]

Change the columns host_is_superhost, instant_bookable, and has_availability into a boolean data type for better data processing:

Replace f and t with False and True
Set the column as type bool
Overwrite the old columns with the new values
[73]
0s
df_list["host_is_superhost"] = df_list["host_is_superhost"].replace({"f":False, "t":True}).astype(bool)
df_list["instant_bookable"] = df_list["instant_bookable"].replace({"f":False, "t":True}).astype(bool)
df_list["has_availability"] = df_list["has_availability"].replace({"f":False, "t":True}).astype(bool)
[74]
0s
df_list.host_is_superhost.head(5)
0    False
1     True
2    False
3    False
4     True
Name: host_is_superhost, dtype: bool
Let's now check to confirm we executed these changes correctly. As seen previously,inspecting the different columns should give you an output that looks something like this:

index	host_is_superhost	instant_bookable	has_availability
0	false	true	true
1	true	false	true
2	false	false	true
3	false	false	true
4	true	false	true
[75]
0s
df_list[["host_is_superhost", "instant_bookable", "has_availability"]].head(5)

Great, you are making a lot of progress! Let's continue!

Task 5: ~~ Float away ~~
[Related section on CoRise]

A closer look at the prices in the four columns price, price_per_person, minimum_price, and service_cost reveals that they all follow the same pattern:

[76]
0s
df_list[["price", "price_per_person", "minimum_price", 'service_cost']].head(5)

All of these four columns have some special characters that you will need to remove before you can change the dtype from object to float.

Remove dollar signs and commas
Convert to float
[77]
0s
df_list["price"] = df_list["price"].str.replace("$", "", regex=True).str.replace(",", "", regex=True).astype("float")
df_list["price_per_person"] = df_list["price_per_person"].str.replace("$", "", regex=True).str.replace(",", "", regex=True).astype("float")
df_list["minimum_price"] = df_list["minimum_price"].str.replace("$", "", regex=True).str.replace(",", "", regex=True).astype("float")
df_list["service_cost"] = df_list["service_cost"].str.replace("$", "", regex=True).str.replace(",", "", regex=True).astype("float")
Let's inspect the different price columns again and see what it look like, expected output should look like the table below.

index	price	price_per_person	minimum_price	service_cost
0	88.0	44.0	176.0	4.99
1	105.0	52.5	315.0	4.99
2	152.0	38.0	304.0	4.99
3	87.0	43.5	174.0	4.99
4	160.0	40.0	320.0	4.99
[78]
0s
df_list[["price", "price_per_person", "minimum_price", 'service_cost']].head(5)

Task 6: Columns with other names
[Related section on CoRise]

The following column names need to be changed:

price into price_in_dollar
neighbourhood_cleansed into neighbourhood
Please finish the code below.

[79]
0s
# Inspect the price and neighbourhood_cleansed columns before renaming
df_list[['price', 'neighbourhood_cleansed']].head()

[80]
0s
df_list = df_list.rename(columns={'price': 'price_in_dollar', 'neighbourhood_cleansed': 'neighbourhood'})
[81]
0s
# Inspect the new columns after renaming
df_list[['price_in_dollar', 'neighbourhood']].head()

Task 7: Categories aren't objects
[Related section on CoRise]

[82]
0s
df_list["neighbourhood"].head(3)
0    IJburg - Zeeburgereiland
1                  Noord-Oost
2                  Noord-West
Name: neighbourhood, dtype: object
[83]
0s
df_list["room_type"].head(3)
0       Private room
1    Entire home/apt
2    Entire home/apt
Name: room_type, dtype: object
Taking a closer look at the neighbourhood and room_type columns reveals that these columns are assigned an object dtype. We want them to be a category dtype. Please set the correct data type below.

[84]
0s
df_list[["neighbourhood", "room_type"]] = df_list[["neighbourhood", "room_type"]].astype("category")
[85]
0s
df_list[["neighbourhood", "room_type"]].info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6165 entries, 0 to 6172
Data columns (total 2 columns):
 #   Column         Non-Null Count  Dtype   
---  ------         --------------  -----   
 0   neighbourhood  6165 non-null   category
 1   room_type      6165 non-null   category
dtypes: category(2)
memory usage: 61.1 KB
We've made quite a few changes 👌, but we're not done yet! Next we need to delete any columns in the DataFrame that we won't use in our analysis.

Task 8: Delete irrelevant columns
[Related section on CoRise]

It might seem intuitive that more data is always better. But that's not always the case. Often in data science you want just the right amount of data — nothing more, nothing less.

We need to delete some columns that are irrelevant to our current use case. Those irrelevant columns are:

host_listings_count
host_total_listings_count
availability_60
availability_90
availability_365
number_of_reviews
number_of_reviews_ltm
reviews_per_month
[86]
0s
# For sanity check, run this cell to inspect the different columns
df_list.columns
Index(['id', 'host_acceptance_rate', 'host_is_superhost',
       'host_listings_count', 'host_total_listings_count', 'neighbourhood',
       'latitude', 'longitude', 'room_type', 'accommodates', 'bedrooms',
       'beds', 'amenities', 'price_in_dollar', 'minimum_nights',
       'maximum_nights', 'has_availability', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
       'review_scores_rating', 'instant_bookable', 'reviews_per_month',
       'price_in_euros', 'price_per_person', 'minimum_price',
       'discount_per_5_days_booked', 'discount_per_10_days_booked',
       'discount_per_30_and_more_days_booked', 'service_cost'],
      dtype='object')
[87]
0s
df_list = df_list.drop(columns=["host_listings_count", "host_total_listings_count", "availability_60", "availability_90", "availability_365", "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month"])
[88]
0s
# Again, run this cell to inspect changes in columns
df_list.columns
Index(['id', 'host_acceptance_rate', 'host_is_superhost', 'neighbourhood',
       'latitude', 'longitude', 'room_type', 'accommodates', 'bedrooms',
       'beds', 'amenities', 'price_in_dollar', 'minimum_nights',
       'maximum_nights', 'has_availability', 'availability_30',
       'number_of_reviews_l30d', 'review_scores_rating', 'instant_bookable',
       'price_in_euros', 'price_per_person', 'minimum_price',
       'discount_per_5_days_booked', 'discount_per_10_days_booked',
       'discount_per_30_and_more_days_booked', 'service_cost'],
      dtype='object')
-- Your Progress --
Let's now have a look at which data types we still need to change and which columns have some null values.

[89]
0s
df_list.info(verbose=True, show_counts=True)
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6165 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    6165 non-null   int64   
 1   host_acceptance_rate                  5365 non-null   float64 
 2   host_is_superhost                     6165 non-null   bool    
 3   neighbourhood                         6165 non-null   category
 4   latitude                              6165 non-null   float64 
 5   longitude                             6165 non-null   float64 
 6   room_type                             6165 non-null   category
 7   accommodates                          6165 non-null   int64   
 8   bedrooms                              5859 non-null   float64 
 9   beds                                  6082 non-null   float64 
 10  amenities                             6165 non-null   int64   
 11  price_in_dollar                       6165 non-null   float64 
 12  minimum_nights                        6165 non-null   int64   
 13  maximum_nights                        6165 non-null   int64   
 14  has_availability                      6165 non-null   bool    
 15  availability_30                       6165 non-null   int64   
 16  number_of_reviews_l30d                6165 non-null   int64   
 17  review_scores_rating                  5581 non-null   float64 
 18  instant_bookable                      6165 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      6165 non-null   float64 
 21  minimum_price                         6165 non-null   float64 
 22  discount_per_5_days_booked            6165 non-null   float64 
 23  discount_per_10_days_booked           6165 non-null   float64 
 24  discount_per_30_and_more_days_booked  6165 non-null   float64 
 25  service_cost                          6165 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 1.1+ MB
We can see from the output above that the columns host_acceptance_rate, review_scores_rating, bedrooms, beds, and price_in_euros still require some processing, as they contain missing values, and/or have dtypes like object or float when they need an integer data type.

To summarize, your cleanup so far has reduced memory usage by almost half, which admittedly given the current amount of memory we are using does not really matter that much. But more importantly, our DataFrame is also more readable now, and we can continue to process it in the next sections.

Task 9: No unique values
[Related section on CoRise]

Let's inspect the price_in_euros column first, because this column seems to contain only null values, which inherently do not add any meaning to our dataset. Remember our unique approach?

[90]
0s
df_list["price_in_euros"].unique()
array([None], dtype=object)
The approach should reveal that this column contains no unique values and is thus empty. Please drop this column.

[91]
0s
df_list.drop(columns=["price_in_euros"])

Task 10: Dropping rows
[Related section on CoRise]

DataFrame info() revealed that some listings have no reviews and an unknown host acceptance rate. Most Airbnb users exclude such listings from their search results. To mimic this filtering approach, please filter out any rows that do not have a review_scores_rating and without a host_acceptance_rate.

A useful method for this approach is the dropna() function.

[92]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6165 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    6165 non-null   int64   
 1   host_acceptance_rate                  5365 non-null   float64 
 2   host_is_superhost                     6165 non-null   bool    
 3   neighbourhood                         6165 non-null   category
 4   latitude                              6165 non-null   float64 
 5   longitude                             6165 non-null   float64 
 6   room_type                             6165 non-null   category
 7   accommodates                          6165 non-null   int64   
 8   bedrooms                              5859 non-null   float64 
 9   beds                                  6082 non-null   float64 
 10  amenities                             6165 non-null   int64   
 11  price_in_dollar                       6165 non-null   float64 
 12  minimum_nights                        6165 non-null   int64   
 13  maximum_nights                        6165 non-null   int64   
 14  has_availability                      6165 non-null   bool    
 15  availability_30                       6165 non-null   int64   
 16  number_of_reviews_l30d                6165 non-null   int64   
 17  review_scores_rating                  5581 non-null   float64 
 18  instant_bookable                      6165 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      6165 non-null   float64 
 21  minimum_price                         6165 non-null   float64 
 22  discount_per_5_days_booked            6165 non-null   float64 
 23  discount_per_10_days_booked           6165 non-null   float64 
 24  discount_per_30_and_more_days_booked  6165 non-null   float64 
 25  service_cost                          6165 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 1.1+ MB
[93]
0s
df_list = df_list.dropna(subset=["review_scores_rating", "host_acceptance_rate"])
[94]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4886 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4886 non-null   int64   
 1   host_acceptance_rate                  4886 non-null   float64 
 2   host_is_superhost                     4886 non-null   bool    
 3   neighbourhood                         4886 non-null   category
 4   latitude                              4886 non-null   float64 
 5   longitude                             4886 non-null   float64 
 6   room_type                             4886 non-null   category
 7   accommodates                          4886 non-null   int64   
 8   bedrooms                              4622 non-null   float64 
 9   beds                                  4817 non-null   float64 
 10  amenities                             4886 non-null   int64   
 11  price_in_dollar                       4886 non-null   float64 
 12  minimum_nights                        4886 non-null   int64   
 13  maximum_nights                        4886 non-null   int64   
 14  has_availability                      4886 non-null   bool    
 15  availability_30                       4886 non-null   int64   
 16  number_of_reviews_l30d                4886 non-null   int64   
 17  review_scores_rating                  4886 non-null   float64 
 18  instant_bookable                      4886 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4886 non-null   float64 
 21  minimum_price                         4886 non-null   float64 
 22  discount_per_5_days_booked            4886 non-null   float64 
 23  discount_per_10_days_booked           4886 non-null   float64 
 24  discount_per_30_and_more_days_booked  4886 non-null   float64 
 25  service_cost                          4886 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 864.6+ KB
Task 11: Reason through your missing data
After setting the right data types, you are often left with making some hard decisions and assumptions about any partially incomplete data in your working dataset. In this case, some beds and bedrooms have no properly assigned values. You can check this by running df_list.info(verbose=True, show_counts=True), which will show that beds and bedrooms have some missing values.

Let's try and make some simple assumptions based on the room_type assigned to the listing. First, inspect which room types are found in the dataset.

[95]
0s
df_list["room_type"].unique()
['Private room', 'Entire home/apt', 'Hotel room', 'Shared room']
Categories (4, object): ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room']
There are four room types. Let's make the assumption that the columns bedrooms and beds are potentially influenced by room_type.

Therefore, we can make the following rules:

If you have a "Private room" or "Shared room" as room_type, then we believe the listing only has one bedroom.
If the listing has "Hotel room" or "Entire home/apt" as room_type, then we can divide the number of guests the listing accomodates by 2 and round up.
If any of these numbers are missing, then we can leave it empty.
Translate these requirements into a Python function, and you get:

def fill_empty_bedrooms(accommodates: int, bedrooms: int, room_type: str) -> int:
    if (room_type == "Private room") or (room_type == "Shared room"):
        return 1
    elif (room_type == "Hotel room") or (room_type == "Entire home/apt"):
        return np.ceil(accommodates / 2)
    else:
        return bedrooms
Let's time this function and see its performance 💪💪! Please run both cells.

[96]
0s
def fill_empty_bedrooms(accommodates: int, bedrooms: int, room_type: str) -> int:
    if (room_type == "Private room") or (room_type == "Shared room"):
        return 1
    elif (room_type == "Hotel room") or (room_type == "Entire home/apt"):
        return np.ceil(accommodates / 2)
    else:
        return bedrooms
[97]
31s
%%timeit -r 4 -n 100

temp_df = df_list.copy()  # Deep copy of the df, not a "view"
temp_df["rooms"] = df_list[["accommodates", "bedrooms", "room_type"]].apply(
    lambda x: fill_empty_bedrooms(x["accommodates"], x["bedrooms"], x["room_type"]),
    axis=1,
)
77 ms ± 20.8 ms per loop (mean ± std. dev. of 4 runs, 100 loops each)
Just like in Week 1, we use timeit to measure the performance of our function. In the case of Pandas, we are using apply() to semi-vectorize our function, but secretly this function just implements something that mimics a for loop. Using a lambda together with apply() allows us to access multiple columns to generate an outcome.

This approach is often good enough, but not always, especially if you are dealing with large datasets. Below we will run the apply() function for output. We've also provided some alternative functions, though they are often found to be slower. (However, it can be a bit of a grey area, so it really depends.)

[98]
0s
# Inspect values in bedrooms before transformation
df_list["bedrooms"].head()
0   1.00
1    NaN
2   1.00
3   1.00
4   2.00
Name: bedrooms, dtype: float64
[99]
0s
df_list2 =df_list.copy()
[100]
0s
df_list["bedrooms"] = df_list[["accommodates", "bedrooms", "room_type"]].apply(
    lambda x: fill_empty_bedrooms(x["accommodates"], x["bedrooms"], x["room_type"]),
    axis=1,
)
[101]
0s
# Inspect values in bedrooms after transformation
df_list["bedrooms"].head()
0   1.00
1   1.00
2   2.00
3   1.00
4   1.00
Name: bedrooms, dtype: float64
[102]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4886 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4886 non-null   int64   
 1   host_acceptance_rate                  4886 non-null   float64 
 2   host_is_superhost                     4886 non-null   bool    
 3   neighbourhood                         4886 non-null   category
 4   latitude                              4886 non-null   float64 
 5   longitude                             4886 non-null   float64 
 6   room_type                             4886 non-null   category
 7   accommodates                          4886 non-null   int64   
 8   bedrooms                              4886 non-null   float64 
 9   beds                                  4817 non-null   float64 
 10  amenities                             4886 non-null   int64   
 11  price_in_dollar                       4886 non-null   float64 
 12  minimum_nights                        4886 non-null   int64   
 13  maximum_nights                        4886 non-null   int64   
 14  has_availability                      4886 non-null   bool    
 15  availability_30                       4886 non-null   int64   
 16  number_of_reviews_l30d                4886 non-null   int64   
 17  review_scores_rating                  4886 non-null   float64 
 18  instant_bookable                      4886 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4886 non-null   float64 
 21  minimum_price                         4886 non-null   float64 
 22  discount_per_5_days_booked            4886 non-null   float64 
 23  discount_per_10_days_booked           4886 non-null   float64 
 24  discount_per_30_and_more_days_booked  4886 non-null   float64 
 25  service_cost                          4886 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 864.6+ KB
Related functions (In general order of preference)
apply(): Apply a function along one or multiple columns
pipe(): Chain multiple transformations/functions after each other
applymap(): Use strictly as a transformation of current value to a new value
itertuples(): Iterate over DataFrame rows as named tuples
iteritems(): Iterate over DataFrame columns
iterrows(): Iterate over DataFrame rows
If you feel like you want to take the function's performance to the next level, we have a bonus (NOT required) question 😱!

(Extra Credit) Task 12: Vectorize!
Can you vectorize the function by using the method described under "Pandas Vectorization", with inspiration from this link? On our hardware, applying vectorization led to results that were at least 30-40x faster. (This might differ a bit on your hardware.)

[103]
0s
temp_df = df_list2.copy()
[104]
0s
temp_df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4886 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4886 non-null   int64   
 1   host_acceptance_rate                  4886 non-null   float64 
 2   host_is_superhost                     4886 non-null   bool    
 3   neighbourhood                         4886 non-null   category
 4   latitude                              4886 non-null   float64 
 5   longitude                             4886 non-null   float64 
 6   room_type                             4886 non-null   category
 7   accommodates                          4886 non-null   int64   
 8   bedrooms                              4622 non-null   float64 
 9   beds                                  4817 non-null   float64 
 10  amenities                             4886 non-null   int64   
 11  price_in_dollar                       4886 non-null   float64 
 12  minimum_nights                        4886 non-null   int64   
 13  maximum_nights                        4886 non-null   int64   
 14  has_availability                      4886 non-null   bool    
 15  availability_30                       4886 non-null   int64   
 16  number_of_reviews_l30d                4886 non-null   int64   
 17  review_scores_rating                  4886 non-null   float64 
 18  instant_bookable                      4886 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4886 non-null   float64 
 21  minimum_price                         4886 non-null   float64 
 22  discount_per_5_days_booked            4886 non-null   float64 
 23  discount_per_10_days_booked           4886 non-null   float64 
 24  discount_per_30_and_more_days_booked  4886 non-null   float64 
 25  service_cost                          4886 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 864.6+ KB
[105]
0s
%%timeit -r 4 -n 100

temp_df["bedrooms"] = temp_df["bedrooms"] 
Case1 = (temp_df["room_type"]=="Private room") | (temp_df["room_type"]=="Shared room")
temp_df.loc[Case1,"bedrooms"] = 1
Case2 = (temp_df["room_type"]=="Hotel room") | (temp_df["room_type"]=="Entire home/apt")
temp_df.loc[Case2,"bedrooms"] = np.ceil(temp_df["accommodates"] / 2)


1.59 ms ± 48.2 µs per loop (mean ± std. dev. of 4 runs, 100 loops each)
[106]
1s
%%timeit -r 4 -n 100

#using faster np.select method
Case1 = (temp_df["room_type"]=="Private room") | (temp_df["room_type"]=="Shared room")
Case2 = (temp_df["room_type"]=="Hotel room") | (temp_df["room_type"]=="Entire home/apt")
temp_df["bedrooms"] = np.select([Case1, Case2],[1, np.ceil(temp_df["accommodates"] / 2)],default=temp_df["bedrooms"])                            
940 µs ± 45.8 µs per loop (mean ± std. dev. of 4 runs, 100 loops each)
Task 13: Clean-up crew
[Related section on CoRise]

Thanks to our logic and assumptions, most listings now have a proper amount of defined rooms. However, there are still a few listings without any number of rooms defined. Remove all rows/entries that have an empty bedrooms, beds.

[107]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4886 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4886 non-null   int64   
 1   host_acceptance_rate                  4886 non-null   float64 
 2   host_is_superhost                     4886 non-null   bool    
 3   neighbourhood                         4886 non-null   category
 4   latitude                              4886 non-null   float64 
 5   longitude                             4886 non-null   float64 
 6   room_type                             4886 non-null   category
 7   accommodates                          4886 non-null   int64   
 8   bedrooms                              4886 non-null   float64 
 9   beds                                  4817 non-null   float64 
 10  amenities                             4886 non-null   int64   
 11  price_in_dollar                       4886 non-null   float64 
 12  minimum_nights                        4886 non-null   int64   
 13  maximum_nights                        4886 non-null   int64   
 14  has_availability                      4886 non-null   bool    
 15  availability_30                       4886 non-null   int64   
 16  number_of_reviews_l30d                4886 non-null   int64   
 17  review_scores_rating                  4886 non-null   float64 
 18  instant_bookable                      4886 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4886 non-null   float64 
 21  minimum_price                         4886 non-null   float64 
 22  discount_per_5_days_booked            4886 non-null   float64 
 23  discount_per_10_days_booked           4886 non-null   float64 
 24  discount_per_30_and_more_days_booked  4886 non-null   float64 
 25  service_cost                          4886 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 993.6+ KB
[108]
0s
df_list = df_list.dropna(subset=["bedrooms", "beds"])
[109]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4817 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4817 non-null   int64   
 1   host_acceptance_rate                  4817 non-null   float64 
 2   host_is_superhost                     4817 non-null   bool    
 3   neighbourhood                         4817 non-null   category
 4   latitude                              4817 non-null   float64 
 5   longitude                             4817 non-null   float64 
 6   room_type                             4817 non-null   category
 7   accommodates                          4817 non-null   int64   
 8   bedrooms                              4817 non-null   float64 
 9   beds                                  4817 non-null   float64 
 10  amenities                             4817 non-null   int64   
 11  price_in_dollar                       4817 non-null   float64 
 12  minimum_nights                        4817 non-null   int64   
 13  maximum_nights                        4817 non-null   int64   
 14  has_availability                      4817 non-null   bool    
 15  availability_30                       4817 non-null   int64   
 16  number_of_reviews_l30d                4817 non-null   int64   
 17  review_scores_rating                  4817 non-null   float64 
 18  instant_bookable                      4817 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4817 non-null   float64 
 21  minimum_price                         4817 non-null   float64 
 22  discount_per_5_days_booked            4817 non-null   float64 
 23  discount_per_10_days_booked           4817 non-null   float64 
 24  discount_per_30_and_more_days_booked  4817 non-null   float64 
 25  service_cost                          4817 non-null   float64 
dtypes: bool(3), category(2), float64(13), int64(7), object(1)
memory usage: 852.4+ KB
Now that we have removed all the empty values, finally we can assign the dtype int instead of float to these two columns.

Please set the columns beds and bedrooms as int.

[126]
0s
df_list["beds"] = df_list["beds"].astype("int")
df_list["bedrooms"] = df_list["bedrooms"].astype("int")
(Extra Credit) Task 14: Speed it up
Lastly, and this is entirely optional, but you can further speed-up data processing by taking the appropriate number of bytes for a given data type, especially when dealing with large datasets (more info here). Currently, the data types are set to 64 bits by default, but most of these could be set to lower values. Good luck 💪💪!

[127]
0s
df_list.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4817 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4817 non-null   int64   
 1   host_acceptance_rate                  4817 non-null   float64 
 2   host_is_superhost                     4817 non-null   bool    
 3   neighbourhood                         4817 non-null   category
 4   latitude                              4817 non-null   float64 
 5   longitude                             4817 non-null   float64 
 6   room_type                             4817 non-null   category
 7   accommodates                          4817 non-null   int64   
 8   bedrooms                              4817 non-null   int64   
 9   beds                                  4817 non-null   int64   
 10  amenities                             4817 non-null   int64   
 11  price_in_dollar                       4817 non-null   float64 
 12  minimum_nights                        4817 non-null   int64   
 13  maximum_nights                        4817 non-null   int64   
 14  has_availability                      4817 non-null   bool    
 15  availability_30                       4817 non-null   int64   
 16  number_of_reviews_l30d                4817 non-null   int64   
 17  review_scores_rating                  4817 non-null   float64 
 18  instant_bookable                      4817 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4817 non-null   float64 
 21  minimum_price                         4817 non-null   float64 
 22  discount_per_5_days_booked            4817 non-null   float64 
 23  discount_per_10_days_booked           4817 non-null   float64 
 24  discount_per_30_and_more_days_booked  4817 non-null   float64 
 25  service_cost                          4817 non-null   float64 
dtypes: bool(3), category(2), float64(11), int64(9), object(1)
memory usage: 852.4+ KB
[ ]


Cleaning Is DONE!


✨ Awesome ✨! You cleaned our data and made sure to watch that the intermediate results aligned with what we expected.

[128]
0s
df_list.info(verbose=True, show_counts=True)
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4817 entries, 0 to 6172
Data columns (total 26 columns):
 #   Column                                Non-Null Count  Dtype   
---  ------                                --------------  -----   
 0   id                                    4817 non-null   int64   
 1   host_acceptance_rate                  4817 non-null   float64 
 2   host_is_superhost                     4817 non-null   bool    
 3   neighbourhood                         4817 non-null   category
 4   latitude                              4817 non-null   float64 
 5   longitude                             4817 non-null   float64 
 6   room_type                             4817 non-null   category
 7   accommodates                          4817 non-null   int64   
 8   bedrooms                              4817 non-null   int64   
 9   beds                                  4817 non-null   int64   
 10  amenities                             4817 non-null   int64   
 11  price_in_dollar                       4817 non-null   float64 
 12  minimum_nights                        4817 non-null   int64   
 13  maximum_nights                        4817 non-null   int64   
 14  has_availability                      4817 non-null   bool    
 15  availability_30                       4817 non-null   int64   
 16  number_of_reviews_l30d                4817 non-null   int64   
 17  review_scores_rating                  4817 non-null   float64 
 18  instant_bookable                      4817 non-null   bool    
 19  price_in_euros                        0 non-null      object  
 20  price_per_person                      4817 non-null   float64 
 21  minimum_price                         4817 non-null   float64 
 22  discount_per_5_days_booked            4817 non-null   float64 
 23  discount_per_10_days_booked           4817 non-null   float64 
 24  discount_per_30_and_more_days_booked  4817 non-null   float64 
 25  service_cost                          4817 non-null   float64 
dtypes: bool(3), category(2), float64(11), int64(9), object(1)
memory usage: 852.4+ KB
Using the function head() reveals the same.

[129]
0s
df_list.head(3)

As you might have noticed, the steps you take and the methods you use to clean your data is very dependent on the data that is given. If you'd like more practice, we encourage you to take a look at this Kaggle tutorial which shows some other problems that might occur when dealing with data.

Mix and Match


As a next step, you will again merge these two datasets, as was shown in this week's content. However, this time around we will take a slightly different angle with the data.

[130]
0s
# The Calendar DataFrame!
df_cal.head(3)

(Extra Credit) Task 15: Minimum stay
[Related section on CoRise]

You are looking to stay at the Airbnb for a minimum of three days, as you think that bookings with a minimum stay of three days are more likely to have discount prices. Since you are unsure which dates you want to book, you'd like to exclude all listing_ids that go below that threshold of three days no matter what time of year.

With these listings excluded, you would like to see the total expected booking price for five days:

Create a list of all unique entries for listing_id that go below the 3-day threshold
Remove them with the provided remove code
Calculate the price of booking a listing for five days by multiplying the current day by 5, and assign this to a column called five_day_dollar_price
[132]
0s
# First start by making a copy for debugging purposes
calendar_newdf = df_cal.copy()

include_list = (
    calendar_newdf[calendar_newdf["minimum_nights"] >= 3]["listing_id"].unique().tolist()
)
[133]
0s
# Get all the listings with a minimum nights of 3+
# Use the include_list
calendar_newdf =  calendar_newdf[calendar_newdf["listing_id"].isin(include_list)]
Related functions
isin(): Filter the DataFrame on provided values
eq(): Filter the DataFrame for all values equal to the provided input
ne(): Filter the DataFrame for all values not equal to the provided input
[134]
0s
calendar_newdf["five_day_dollar_price"] = calendar_newdf["price_in_dollar"] * 5
[136]
0s
calendar_newdf.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1363640 entries, 365 to 2252049
Data columns (total 7 columns):
 #   Column                 Non-Null Count    Dtype         
---  ------                 --------------    -----         
 0   listing_id             1363640 non-null  int64         
 1   date                   1363640 non-null  datetime64[ns]
 2   available              1363640 non-null  bool          
 3   price_in_dollar        1363640 non-null  float64       
 4   minimum_nights         1363640 non-null  int64         
 5   maximum_nights         1363640 non-null  int64         
 6   five_day_dollar_price  1363640 non-null  float64       
dtypes: bool(1), datetime64[ns](1), float64(2), int64(3)
memory usage: 74.1 MB
Now let's transform our newly created DataFrame into a pivot table, where we aggregate our rows using the listing_id as the index, and the columns available and five_day_dollar_price as values.

[137]
0s
calendar_summarizeddf = pd.pivot_table(
    data=calendar_newdf,
    index=["listing_id"],
    values=["available", "five_day_dollar_price"],
    aggfunc=np.mean,  # The default aggregation function used
    # for merging multiple related rows of data.
)

calendar_summarizeddf.head(3)

(Extra Credit) Task 16: Maximum price and date
[Related section on CoRise]

Can you make a pivot table that states the maximum price_in_dollar for every Airbnb listing?

The expected pivot table output should look like.

listing_id	price_in_dollar
2818	80.0
44391	240.0
49552	300.0
[138]
0s
temp_sum_df = pd.pivot_table(
    data = calendar_newdf,
    index=['listing_id'],
    values=['price_in_dollar'],
    aggfunc=np.max
)


temp_sum_df.head(3)

Join Us?!


We are going to merge the pivot table that includes the five_day_dollar_price for each listing with listings_df. We have to keep in mind that we want to keep only those rows of Airbnb listing IDs that are present in both datasets.

Task 17: Mergin'
[Related section on CoRise]

Let's use the pd.merge() operation as was shown on CoRise, with the pivot table on the right and the Listings DataFrame on the left. Make sure to provide which columns you want to join on for our pivot table and the DataFrame.

The expected merged table should look same as

index	id	host_acceptance_rate	host_is_superhost	neighbourhood	latitude	longitude	room_type	accommodates	bedrooms	beds	amenities	price_in_dollar	minimum_nights	maximum_nights	has_availability	availability_30	number_of_reviews_l30d	review_scores_rating	instant_bookable	price_per_person
0	35815036	1.0	true	Noord-Oost	52.42419	4.95689	Entire home/apt	2	1	1	5	105.0	3	100	true	4	6	4.96	false	52.5
1	19572024	1.0	false	Watergraafsmeer	52.30739	4.90833	Entire home/apt	6	3	6	14	279.0	3	300	true	6	3	4.69	false	46.5
2	2973384	0.38	false	Watergraafsmeer	52.30989	4.90528	Entire home/apt	5	3	3	7	185.0	6	21	true	0	0	4.83	false	37.0
[139]
0s
final_df = pd.merge(
     df_list,
    calendar_summarizeddf,
    left_on=['id'],
    right_on=['listing_id'],
    how='inner'
)
final_df.head(3)

(Extra Credit) Task 18: Groups are great
[Related section on CoRise]

Now, let's perform a groupby where we look at the median values of five_day_dollar_price and review_scores_accuracy with respect to the room_type. Do these results match your intuition?

The expected group by table should look same as,

room_type	review_scores_rating	five_day_dollar_price
Entire home/apt	4.88	972.7397260273973
Hotel room	4.5649999999999995	908.1575342465753
Private room	4.79	681.986301369863
Shared room	4.6	565.027397260274
[140]
0s
final_df.groupby(by=['room_type'])[
    [
        'review_scores_rating',
        'five_day_dollar_price'
    ]

].median()

You might have expected that shared rooms are the cheapest and thus have the lowest rating with respect to median scores. The same can't be said for the most expensive option — a hotel room. Will this influence your future considerations when booking 🤔?

(But before you let this influence your decisions too much, it might be better to assume that this data might be biased in favor of Airbnb and not hotels in general. 🤷)

You've walked through all the most important parts of Pandas. It's a really easy-to-use library that shares a lot of syntax with NumPy. It is great for analyzing and cleaning datasets, and as you might have discovered with the previous code, Pandas allows you to really go into the nitty-gritty details of your dataset. These skills are invaluable for a data scientist, and will empower you to utilize data where you work now, or even where you could work in the future!

The next steps involve downloading the files to your local computer so that you can make an app for your portfolio. After, we will provide some suggestions on how you can extend this project, along with some interesting links to investigate.

Download the Dataset to Your Local Machine
Let's first export our final DataFrame.
