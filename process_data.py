from typing import List
import streamlit as st
import warnings
import polars as pl
from collections import Counter
import string
import re

from config import Config

warnings.filterwarnings('ignore')


@st.cache_data
def join_descriptions(descriptions, stopwords, most_common_n=1000):
    """Cache the operation of joining descriptions into a single string."""
    all_words = []
    config = Config()
    pattern = r'[0-9]'

    # Match all digits in the string and replace them with an empty string

    words = [re.sub(pattern, '', description.translate(str.maketrans('', '', string.punctuation))).split(" ")[0] for description in descriptions]
    replaced_words = [config.replace_dict[word.strip().lower()] if word.strip().lower() in config.replace_dict else word for word in words]
    filtered_words = [word.strip().lower() for word in replaced_words if word.strip().lower() not in stopwords and word != ""]


    counter = Counter(filtered_words)
    most_common_words = counter.most_common(most_common_n)

    # Combine the most common words into a single string for word cloud generation
    text = ' '.join([word for word, _ in most_common_words])
    return text, most_common_words


class ProcessData:
    def __init__(self):
        self.config = Config()

    def __read_data(self) -> pl.DataFrame:
        df = pl.read_csv(
            self.config.input_path.joinpath(self.config.csv_file_name),
            null_values=["NA"],
        )
        df = df.drop(self.config.ignore_columns)
        return df

    @staticmethod
    def convert_columns(df: pl.DataFrame):
        df_converted = df.with_columns(
            pl.col("date").str.to_date(format="%b%y").alias("date"),
            pl.col('yearConstructed').cast(pl.Int32),
            pl.col("totalRent").str.replace(",", ".").cast(pl.Float64)


        )
        return df_converted

    @staticmethod
    def replace_values(df: pl.DataFrame) -> pl.DataFrame:
        # replaces values
        df_cleaned = df.with_columns(
            pl.col("heatingType").str.replace_all(r"_", " "),
            pl.col("regio1").str.replace_all(r"_", " ").alias("region"),
            pl.col("firingTypes").str.replace_all(r"_", " "),
            pl.col("condition").str.replace_all(r"_", " "),
            pl.col("typeOfFlat").str.replace_all(r"_", " "),
            pl.col("energyEfficiencyClass").str.replace_all(r"_", " "),
        )
        df_cleaned= df_cleaned.drop("regio1")
        return df_cleaned

    def get_clean_data(self) -> pl.DataFrame:
        df = self.__read_data()
        df_cleaned = self.replace_values(df=df)
        df_cleaned = self.convert_columns(df=df_cleaned)
        df_cleaned = df_cleaned.filter(
            pl.col("yearConstructed") <= 2020,
            pl.col("yearConstructed") > 1930,
            pl.col("baseRent")
            < pl.col("baseRent").mean()
            + (0.095 * pl.col("baseRent").std()),  # 1 -> num of std
            pl.col("baseRent") > 100,
        ).unique(subset=list(df_cleaned.columns), maintain_order=True)
        return df_cleaned

    @staticmethod
    def remove_none_from_list(list_element: List) -> List:
        cleaned_list = [element for element in list_element if element is not None]
        return cleaned_list

    @staticmethod
    def filter_data(df: pl.DataFrame, date_range, location, property_type, energy_type):
        filtered_df = df

        # Filter by date range if provided
        if date_range:
            if isinstance(date_range, tuple):
                if len(date_range) == 1:
                    # If only one date is selected, use the same date for start and end to avoid unpacking error
                    start_date, end_date = date_range[0], date_range[0]
                    filtered_df = filtered_df.filter(
                        pl.col("date").is_between(start_date, end_date)
                    )
                elif len(date_range) == 2:

                    # If two dates are selected, unpack them normally
                    start_date, end_date = date_range
                    filtered_df = filtered_df.filter(
                        pl.col("date").is_between(start_date, end_date)
                    )

            else:
                # If date_range somehow isn't a list (shouldn't happen with date_input), use default dates
                start_date, end_date = df["date"].min(), df["date"].max()
                filtered_df = filtered_df.filter(
                    pl.col("date").is_between(start_date, end_date)
                )

        # Filter by location if not 'All'
        if location and len(location) > 0 and location != "All":

            filtered_df = filtered_df.filter(pl.col("region") == location)

        # Filter by location if not 'All'
        if energy_type and len(energy_type) > 0 and energy_type != "All":

            filtered_df = filtered_df.filter(
                pl.col("energyEfficiencyClass") == energy_type
            )

        # Filter by property type if not 'All'
        if len(property_type) > 0 and property_type != "All":

            filtered_df = filtered_df.filter(pl.col("typeOfFlat").is_in(property_type))

        return filtered_df

    @staticmethod
    def format_pandas_dataframe(df):
        # Format yearConstructed as integer and convert to string to avoid any alteration
        df['yearConstructed'] = df['yearConstructed'].astype(int).astype(str)

        # Format totalRent with dot as decimal separator and convert to string
        df['totalRent'] = df['totalRent'].apply(lambda x: f"{x:.2f}".replace(',', '.'))
        df['baselRent'] = df['baseRent'].apply(lambda x: f"{x:.2f}".replace(',', '.'))

        df = df.replace('nan', None)
        return df

    @staticmethod
    def filter_table(df, rent_range, year_range, space_range, sort_column) -> pl.DataFrame:
        filtered_df = df.filter([
            (pl.col("baseRent") >= rent_range[0]) & (pl.col("baseRent") <= rent_range[1]),
            (pl.col("yearConstructed") >= year_range[0]) & (pl.col("yearConstructed") <= year_range[1]),
            (pl.col("livingSpace") >= space_range[0]) & (pl.col("livingSpace") <= space_range[1])
        ])
        if sort_column != "None":
            filtered_df = filtered_df.sort(sort_column)
        else:
            filtered_df = filtered_df
        return filtered_df.to_pandas()

    @staticmethod
    def categorize_age(age, bins):
        for i, b in enumerate(bins):
            if age <= b:
                return f"{bins[i - 1] if i > 0 else 0}-{b}"
        return f"{bins[-1]}+"

    @staticmethod
    def categorize_rent(rent, bins):
        for i, b in enumerate(bins):
            if rent <= b:
                return f"{bins[i - 1] if i > 0 else 0}-{b}"
        return f"{bins[-1]}+"

    @staticmethod
    def sort_key(x):
        # Check if the category represents an overflow category, e.g., '9999800+'
        if str(x).endswith("+"):
            # Return a large number to ensure it sorts last
            return float("inf")
        return float(str(x).split("-")[0])

    # Function to retrieve latitude for a given region

    def get_latitude(self, region):
        lat = self.config.region_coordinates.get(region, {}).get("lat", None)

        return lat

    # Function to retrieve longitude for a given region
    def get_longitude(self, region):
        lon = self.config.region_coordinates.get(region, {}).get("lon", None)

        return lon
