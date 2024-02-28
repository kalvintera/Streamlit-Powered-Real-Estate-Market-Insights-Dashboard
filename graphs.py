import calendar
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objs
import plotly_express as px
import plotly.graph_objects as go
import polars as pl
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

from typing import Tuple

from process_data import ProcessData, join_descriptions
from config import Config


class CreateGraphs:

    def __init__(self):
        self.process = ProcessData()
        self.config = Config()
        self.table_style = """
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                th {
                    background-color: #f0f0f0;
                }
                </style>
                """
        self.german_stopwords = stopwords.words('german')

    def get_overview_info(self, main_df: pd.DataFrame):

        for column in self.config.categorical_column_names:
            main_df[column] = main_df[column].astype('category')
        # Calculate statistics
        num_variables = main_df.shape[1]
        num_observations = main_df.shape[0]
        missing_cells = main_df.isnull().sum().sum()
        missing_cells_percent = (
            missing_cells / (main_df.shape[0] * main_df.shape[1])
        ) * 100
        duplicate_rows = main_df.duplicated().sum()
        duplicate_rows_percent = (duplicate_rows / main_df.shape[0]) * 100
        total_size_in_memory = main_df.memory_usage(deep=True).sum()
        average_record_size = total_size_in_memory / num_observations

        # Variable types count
        categorical_count = sum(
            1 for dtype in main_df.dtypes if isinstance(dtype, pd.CategoricalDtype)
        )
        numeric_count = sum(
            pd.api.types.is_numeric_dtype(dtype) for dtype in main_df.dtypes
        )
        boolean_count = sum(
            pd.api.types.is_bool_dtype(dtype) for dtype in main_df.dtypes
        )
        text_count = sum(
            pd.api.types.is_string_dtype(dtype) for dtype in main_df.dtypes if not isinstance(dtype, pd.CategoricalDtype)
        )
        datetime_count = sum(
            pd.api.types.is_datetime64_any_dtype(dtype) for dtype in main_df.dtypes
        )

        # Create a list of dictionaries for each row in the table
        data_overview = [
            {"Metric": "Number of variables", "Value": num_variables},
            {"Metric": "Number of observations", "Value": num_observations},
            {"Metric": "Missing cells", "Value": missing_cells},
            {"Metric": "Missing cells (%)", "Value": f"{missing_cells_percent:.2f}%"},
            {"Metric": "Duplicate rows", "Value": duplicate_rows},
            {"Metric": "Duplicate rows (%)", "Value": f"{duplicate_rows_percent:.2f}%"},
            {
                "Metric": "Total size in memory",
                "Value": f"{total_size_in_memory / (1024 ** 2):.2f} MiB",
            },
            {
                "Metric": "Average record size in memory",
                "Value": f"{average_record_size:.1f} B",
            },
            {"Metric": "Categorical", "Value": categorical_count},
            {"Metric": "Numeric", "Value": numeric_count},
            {"Metric": "Boolean", "Value": boolean_count},
            {"Metric": "Text", "Value": text_count},
            {"Metric": "DateTime", "Value": datetime_count},
        ]

        # Convert the list of dictionaries to a markdown table string without column lines
        # Start the HTML table
        table_html = "<table>"

        # Add the header row
        table_html += "<tr><th>Metric</th><th>Value</th></tr>"

        # Add data rows
        for item in data_overview:
            table_html += f"<tr><td>{item['Metric']}</td><td>{item['Value']}</td></tr>"

        # Close the table HTML tag
        table_html += "</table> <br>"

        # Combine the styling with the table
        full_html = self.table_style + table_html
        return full_html

    @staticmethod
    def calculate_key_metrics(filtered_df):

        median_rent = filtered_df["baseRent"].median()
        num_properties = len(filtered_df)

        # Example: Calculate average year of construction for properties
        avg_year_constructed = filtered_df["yearConstructed"].median()

        return median_rent, num_properties, avg_year_constructed

    @staticmethod
    def average_rent_year_distribution(
        df: pl.DataFrame,
    ) -> Tuple[plotly.graph_objs.Figure, plotly.graph_objs.Figure]:
        # Create a histogram with Plotly
        # Create a histogram with additional features
        fig_rent = px.histogram(df, x="baseRent")
        # Adjust bin size to 100 euros
        fig_rent.update_traces(
            xbins=dict(start=df["baseRent"].min(), end=df["baseRent"].max(), size=250)
        )

        # Customize the layout if needed
        fig_rent.update_layout(
            title="Distribution of Base Rent with Rug Plot",
            xaxis_title="Base Rent",
            yaxis_title="Count",
        )

        # Plotting the histogram for yearConstructed
        fig_year = px.histogram(
            df, x="yearConstructed", title="Distribution of Year Constructed"
        )
        fig_year.update_traces(
            xbins=dict(
                start=df["yearConstructed"].min(),
                end=df["yearConstructed"].max(),
                size=1,
            )
        )
        # Each bin is one year

        return fig_rent, fig_year

    @staticmethod
    def create_rent_trend_over_time(df: pl.DataFrame) -> Tuple:
        """

        :param df: filtered dataframe based on input
        :return: a plotly figure/line chart to be rendered and aggregated df
        """
        # Aggregate data to get average rent by date
        aggregated_df = (
            df.groupby("date")
            .agg(
                [
                    pl.col("baseRent").median().alias("Average Base Rent"),
                    pl.col("totalRent").median().alias("Average Total Rent"),
                ]
            )
            .sort("date")
        )

        # Convert to Pandas DataFrame for Plotly
        aggregated_df_pd = aggregated_df.to_pandas()

        # Plotting with Plotly
        fig = px.line(
            aggregated_df_pd,
            x="date",
            y=["Average Base Rent", "Average Total Rent"],
            title="Rent Trend Analysis Over Time",
            labels={"value": "Average Rent (€)", "variable": "Rent Type"},
            markers=True,
        )

        return fig, aggregated_df

    @staticmethod
    def create_rent_trend_over_time_by_type(df: pl.DataFrame) -> Tuple:
        """
        :param df: Filtered dataframe based on input
        :return: A plotly figure/line chart to be rendered and aggregated df
        """
        # Aggregate data to get average rent by date and type
        aggregated_df = (
            df.groupby(["date", "typeOfFlat"])
            .agg(
                [
                    pl.col("baseRent").median().alias("Average Base Rent"),
                    pl.col("totalRent").median().alias("Average Total Rent"),
                ]
            )
            .sort(["date", "typeOfFlat"])
        )

        # Convert to Pandas DataFrame for Plotly
        aggregated_df_pd = aggregated_df.to_pandas()

        # Plotting with Plotly: Create a line chart for each property type
        fig = px.line(
            aggregated_df_pd,
            x="date",
            y="Average Base Rent",
            color="typeOfFlat",
            title="Rent Trend Analysis Over Time by Property Type",
            labels={
                "value": "Average Rent (€)",
                "date": "Date",
                "typeOfFlat": "Property Type",
            },
            markers=True,
            template="plotly_white",
        )

        # Enhance the figure's readability and professional appearance
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Rent (€)",
            legend_title="Property Type",
            font=dict(family="Arial, sans-serif", size=12, color="#7f7f7f"),
            hovermode="x unified",
        )

        # Highlight types by adding a hover template for more info
        fig.update_traces(
            hovertemplate="Type: %{color} <br>Date: %{x} <br>Average Base Rent: %{y}€"
        )

        # Optionally, customize the legend and lines for clarity and aesthetics
        fig.update_traces(mode="lines+markers", line=dict(width=2))
        fig.update_layout(
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(family="Courier", size=12, color="black"),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2,
            )
        )

        return fig, aggregated_df

    @staticmethod
    def get_comparison_rent_by_property(df: pl.DataFrame) -> plotly.graph_objs.Figure:
        # Aggregate data to get average rent by property type
        aggregated_df = (
            df.groupby("typeOfFlat")
            .agg([pl.median("baseRent").round().alias("Average Rent")])
            .sort("Average Rent")
        )

        # Convert to Pandas DataFrame for Plotly visualization
        aggregated_df_pd = aggregated_df.to_pandas()

        # Plotting with Plotly
        fig = px.bar(
            aggregated_df_pd,
            x="typeOfFlat",
            y="Average Rent",
            title="Average Rent by Property Type",
            labels={"typeOfFlat": "Property Type", "Average Rent": "Average Rent (€)"},
            color="Average Rent",  # Color bars by average rent
            template="plotly_white",  # Use a clean and professional template
            text="Average Rent",
        )  # Show average rent on bars

        # Customize the layout for a more professional look
        fig.update_layout(
            xaxis_title="Property Type",
            yaxis_title="Average Rent (€)",
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        )

        # Customize bar appearance
        fig.update_traces(
            marker_line_color="rgb(8,48,107)",  # Border line color of bars
            marker_line_width=1.5,  # Border line width of bars
            opacity=0.8,
        )  # Bar opacity

        return fig

    @staticmethod
    def get_comparison_rent_by_season(df: pl.DataFrame) -> Tuple:

        df = df.with_columns(pl.col("date").dt.month().alias("month"))
        monthly_avg_rent = (
            df.groupby("month")
            .agg([pl.col("baseRent").median().alias("average_baseRent")])
            .sort("month")
        )

        # Convert the aggregated DataFrame to Pandas for visualization, if preferred
        monthly_avg_rent_pd = monthly_avg_rent.to_pandas()

        # Create a line chart
        fig = px.line(
            monthly_avg_rent_pd,
            x="month",
            y="average_baseRent",
            title="Seasonal Variations in Average Rent Prices",
            labels={"month": "Month", "average_baseRent": "Average Base Rent (€)"},
            markers=True,
        )

        # Create a mapping of month numbers to month names
        month_names = {i: name for i, name in enumerate(calendar.month_abbr) if i}

        # Assuming 'monthly_avg_rent' is your aggregated DataFrame with a 'month' column
        unique_months = monthly_avg_rent["month"].unique().to_list()
        unique_months.sort()  # Ensure months are in chronological order

        # Generate tick values and text dynamically based on unique months in the data
        tickvals = unique_months
        ticktext = [month_names[month] for month in unique_months]

        # Apply dynamic tick values and text based on the actual data
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

        return fig, monthly_avg_rent_pd

    @staticmethod
    def get_impact_prop_features_on_rent(df: pl.DataFrame) -> Tuple:
        features = ["balcony", "garden", "noParkSpaces", "cellar", "lift"]
        plot_data = {
            "Feature": [],
            "Condition": [],
            "Average Base Rent": [],
            "Count": [],
            "P-Value": [],  # Storing P-values for statistical significance
        }

        for feature in features:
            if not df.is_empty():
                if feature == "noParkSpaces":
                    df = df.with_columns(
                        pl.col("noParkSpaces").cast(pl.Boolean)
                    )  # Ensure correct data type

                # Filtering properties based on feature presence/absence
                group_yes = df.filter(pl.col(feature) == True)["baseRent"].to_numpy()
                group_no = df.filter(pl.col(feature) == False)["baseRent"].to_numpy()

                # Perform t-test
                _, p_value = ttest_ind(
                    group_yes, group_no, equal_var=False, nan_policy="omit"
                )

                # Calculate medians and counts
                median_yes = np.median(group_yes)
                median_no = np.median(group_no)
                count_yes = len(group_yes)
                count_no = len(group_no)

                # Appending data
                plot_data["Feature"].extend([feature, feature])
                plot_data["Condition"].extend(["With", "Without"])
                plot_data["Average Base Rent"].extend([median_yes, median_no])
                plot_data["Count"].extend([count_yes, count_no])
                plot_data["P-Value"].extend(
                    [p_value, p_value]
                )  # Append p-value for both conditions

        # Convert plot data to DataFrame for plotting outside the loop
        plot_df = pl.DataFrame(plot_data)

        # Create an advanced bar chart
        fig = px.bar(
            plot_df.to_pandas(),
            x="Feature",
            y="Average Base Rent",
            color="Condition",
            title="Impact of Property Features on Average Base Rent",
            labels={
                "Average Base Rent": "Average Base Rent (€)",
                "Feature": "Property Feature",
                "Condition": "Condition",
            },
            hover_data=["Count", "P-Value"],  # Include count and p-value in hover data
            barmode="group",
        )

        # Customize the chart
        fig.update_layout(
            xaxis_title="Property Feature",
            yaxis_title="Average Base Rent (€)",
            legend_title="Feature Presence",
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            template="plotly_white",
        )

        # Highlight statistical significance in the hover template
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Condition: %{marker.color}<br>Average Base Rent: €%{y:.2f}"
            "<br>Count: %{customdata[0]}<br>P-Value: %{customdata[1]:.3f}"
        )

        return fig, plot_df

    def generate_age_rent_heatmap(self, df: pl.DataFrame) -> Tuple:
        current_year = datetime.now().year

        # Calculate the age of the properties
        df = df.with_columns((current_year - pl.col("yearConstructed")).alias("Age"))

        # Define age and rent bins
        age_bins = [0, 5, 10, 20, 30, 50, 100, 150]
        rent_bins = range(
            0, int(df["baseRent"].max()) + 1, 200
        )  # Adjust step based on your data distribution

        # Apply categorization
        df = df.with_columns(
            [
                pl.col("Age")
                .apply(lambda x: self.process.categorize_age(x, age_bins))
                .alias("Age Group"),
                pl.col("baseRent")
                .apply(lambda x: self.process.categorize_rent(x, rent_bins))
                .alias("Rent Range"),
            ]
        )

        # Aggregate data
        heatmap_data = df.groupby(["Age Group", "Rent Range"]).agg(
            pl.count().alias("Count")
        )
        # Convert to Pandas DataFrame for pivot
        heatmap_df = heatmap_data.to_pandas()
        # Pivot the DataFrame
        pivot_df = heatmap_df.pivot(
            index="Rent Range", columns="Age Group", values="Count"
        ).fillna(0)

        # Sorting the index and columns based on the numerical part of the categorization
        # Adjusted sorting logic
        pivot_df.index = sorted(pivot_df.index, key=self.process.sort_key)
        pivot_df.columns = sorted(pivot_df.columns, key=self.process.sort_key)
        # Create the heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(
                x="Property Age Group",
                y="Rent Price Range (€)",
                color="Number of Properties",
            ),
            x=pivot_df.columns,
            y=pivot_df.index,
            aspect="auto",
        )

        fig.update_xaxes(side="top")

        return fig, pivot_df

    def generate_affordability_graph(self, df: pl.DataFrame):
        # Add a column for average household income per capita based on 'regio1'

        income_df = pl.DataFrame(
            {
                "region": list(self.config.income_mapping.keys()),
                "Average Income": list(self.config.income_mapping.values()),
            }
        )

        # Assuming 'df' is your original DataFrame with a 'regio1' column
        df = df.join(income_df, on="region", how="left")

        # Calculate rent-to-income ratio (assuming 'baseRent' is monthly rent)
        # And assuming the income is annual, we convert it to monthly by dividing by 12
        df = df.with_columns(
            (pl.col("baseRent") / (pl.col("Average Income") / 12) * 100).alias(
                "Rent-to-Income Ratio"
            )
        )
        # Calculate average rent-to-income ratio by region
        avg_ratio_by_region = (
            df.groupby("region")
            .agg(
                [
                    pl.col("Rent-to-Income Ratio")
                    .median()
                    .alias("Average Rent-to-Income Ratio")
                ]
            )
            .sort("Average Rent-to-Income Ratio")
        )

        # Convert to Pandas DataFrame for Plotly (optional, for convenience if using Plotly)
        avg_ratio_by_region_df = avg_ratio_by_region.to_pandas()

        # Create a bar chart
        fig = px.bar(
            avg_ratio_by_region_df,
            x="region",
            y="Average Rent-to-Income Ratio",
            title="Average Rent-to-Income Ratio by Region",
            labels={
                "region": "Region",
                "Average Rent-to-Income Ratio": "Average Rent-to-Income Ratio (%)",
            },
            color="Average Rent-to-Income Ratio",
            color_continuous_scale=px.colors.sequential.Viridis,
        )

        # Generate tick values and text dynamically based on unique months in the data
        tickvals = list(avg_ratio_by_region_df["region"].unique())
        ticktext = [
            region for region in list(avg_ratio_by_region_df["region"].unique())
        ]

        # Apply dynamic tick values and text based on the actual data
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

        return fig, avg_ratio_by_region_df

    def create_dynamic_map(self, df: pl.DataFrame):
        # Convert dictionary to include 'Region' as a key for easy mapping

        # Aggregate data to compute the count of properties and median rental prices per region
        aggregated_df = df.groupby("region").agg(
            [
                pl.count().alias("CountOfProperties"),
                pl.col("baseRent").median().alias("MedianRent"),
            ]
        )

        aggregated_df = aggregated_df.with_columns(
            [
                pl.col("region").apply(self.process.get_latitude).alias("Latitude"),
                pl.col("region").apply(self.process.get_longitude).alias("Longitude"),
            ]
        )

        # Convert to pandas DataFrame for Plotly visualization
        final_df_pd = aggregated_df.to_pandas()

        # Generating the map
        fig = px.scatter_mapbox(
            final_df_pd,
            lat="Latitude",
            lon="Longitude",
            size="CountOfProperties",
            color="MedianRent",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=5,
            mapbox_style="carto-positron",
            hover_name="region",
            hover_data={"MedianRent": True, "CountOfProperties": True},
        )

        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        return fig, final_df_pd

    @staticmethod
    def plot_energy_efficiency_impact(df: pl.DataFrame):
        # Filter data based on selections
        # Aggregate data to get median rent and count of properties by energy efficiency class
        aggregated_df = (
            df.groupby("energyEfficiencyClass")
            .agg(
                [
                    pl.col("baseRent").median().alias("MedianBaseRent"),
                    pl.count("energyEfficiencyClass").alias("PropertyCount"),
                ]
            )
            .sort("energyEfficiencyClass")
        )  # Ensure classes are sorted for better visualization

        # Convert to Pandas DataFrame for plotting
        aggregated_df_pd = aggregated_df.to_pandas()

        # Creating figure with secondary y-axis
        fig = go.Figure()

        # Add Median Base Rent Bar
        fig.add_trace(
            go.Bar(
                x=aggregated_df_pd["energyEfficiencyClass"],
                y=aggregated_df_pd["MedianBaseRent"],
                name="Median Base Rent",
            )
        )

        # Add Property Count Line
        fig.add_trace(
            go.Scatter(
                x=aggregated_df_pd["energyEfficiencyClass"],
                y=aggregated_df_pd["PropertyCount"],
                name="Property Count",
                mode="lines+markers",
                yaxis="y2",
            )
        )

        # Customize appearance
        fig.update_layout(
            title_text="Median Base Rent and Property Count by Energy Efficiency Class",
            xaxis_title_text="Energy Efficiency Class",
            yaxis_title_text="Median Base Rent (€)",
            legend_title="Metrics",
            template="plotly_white",
        )

        # Set y-axes titles
        fig.update_layout(
            yaxis=dict(title="Median Base Rent (€)"),
            yaxis2=dict(overlaying="y", side="right"),
        )

        # Improve readability and professional appearance
        fig.update_layout(font=dict(family="Arial, sans-serif", size=12, color="#333"))

        return fig

    def generate_wordcloud(self, df, max_words):
        descriptions = df.select(pl.col("description").drop_nans().drop_nulls())['description'].to_list()

        processed_text, most_occur = join_descriptions(
            descriptions=descriptions,
            stopwords=self.german_stopwords + self.config.additional_stopwords,  # Define your list of German stopwords

        )

        # Generate wordcloud with adjusted parameters
        wordcloud = WordCloud(stopwords=self.german_stopwords + self.config.additional_stopwords,
                              max_words=max_words, background_color='white',
                              width=800, height=600, font_step=1).generate(processed_text)

        # Display with higher quality
        fig_wordcloud, ax = plt.subplots(figsize=(10, 7), dpi=120)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        # -------barchart ###############

        # Convert most_occur to a DataFrame for easier handling
        df_most_occur = pd.DataFrame(most_occur, columns=['Word', 'Frequency'])

        # Sort the data for better visualization
        df_most_occur_sorted = df_most_occur.sort_values('Frequency', ascending=False)

        # Create the bar chart with a specified figure height and dynamic x-axis range
        fig_barchart = px.bar(
            df_most_occur_sorted.head(20),  # Limit to 'max_words' number of bars
            y='Word', x='Frequency', orientation='h',
            color='Frequency',  # Color scale based on frequency
            labels={'Frequency': 'Frequency', 'Word': 'Word'},  # Customize axis labels
            title='Most Frequent Words',  # Chart title
            height=600  # Set a specific height for the figure
        )

        # Improve layout and set the x-axis range dynamically based on the max frequency value
        fig_barchart.update_layout(
            xaxis_title='Frequency',
            yaxis_title='Word',
            title={
                'text': "Most Frequent Words",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template='plotly_white',  # Use a clean template
            coloraxis_colorbar=dict(
                title='Frequency'  # Color bar title
            ),
            xaxis_range=[0, df_most_occur_sorted['Frequency'].max() + df_most_occur_sorted['Frequency'].max() * 0.1]
            # Increase the range by 10% for aesthetics
        )

        return fig_wordcloud, fig_barchart
