import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from config import Config
from process_data import ProcessData
from graphs import CreateGraphs
from dynamic_insights import CreateInsights


# set up config
config = Config()
process = ProcessData()
graphs = CreateGraphs()
insights = CreateInsights()

st.set_page_config(layout="wide")
# ----------------------------
main_df = process.get_clean_data()

# Sidebar widgets for dynamic interaction
st.sidebar.header("Filters")

# Date Range Preparation
earliest_date, latest_date = main_df["date"].min(), main_df["date"].max()

# Location Filter Preparation
location_options = ["All"] + main_df["region"].unique().to_list()
location_options = sorted(location_options)

# Property Type Preparation
prop_types = ["All"] + process.remove_none_from_list(
    main_df["typeOfFlat"].unique().to_list()
)
prop_types = sorted(prop_types)

# Energy Efficiency Filter Preparation
energy_types = ["All"] + main_df["energyEfficiencyClass"].unique().to_list()
location_options = sorted(location_options)

# Date Filter
date_range = st.sidebar.date_input("Date Range", [earliest_date, latest_date])

# Location Filter
selected_location = st.sidebar.selectbox(
    "Location", index=None, options=location_options
)

# Property Type Filter
selected_prop_types = st.sidebar.multiselect(
    "Select Property Type",
    options=tuple(prop_types),
)
# Energy Efficiency Class Filter
selected_efficiency_class = st.sidebar.selectbox(
    "Select Energy Efficiency Class", options=["All"] + energy_types
)

# Apply filters
filtered_data = process.filter_data(
    main_df,
    date_range,
    selected_location,
    selected_prop_types,
    selected_efficiency_class,
)

with st.sidebar:
    add_vertical_space(10)
    st.sidebar.markdown(
        "<small style='margin-top:15px;'>made by SteinCode</small> ",
        unsafe_allow_html=True,
    )

# Create columns to center elements (image and title)
img_1_co, img_2_co = st.columns([2, 5])
with img_2_co:
    st.image(config.new_image)
left_co, cent_co, last_co = st.columns([2, 5, 1])


with cent_co:  # This ensures that content within this block is centered

    st.markdown(
        "<h1 style='text-align: center; margin-top: -3%; font-size: 28px;'>Rental Property Analysis</h1>",
        unsafe_allow_html=True,
    )

# Introductory text centered using the columns, but now proceeding without columns
st.write(
    "Access detailed rental property data for analysis and controlling tasks with our app. "
    "It provides data exploration tools to analyze rent, property features, and geographic details. "
    "The application supports trend analysis and financial monitoring, offering insights for decision-making "
    "without complex terminology."
)

st.subheader("Data Overview")


# -------- TABLE ------------
@st.cache_data(show_spinner=True)
def split_frame(input_df, rows):
    df = [input_df.loc[i: i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df


with st.expander("**Filter Table**"):
    slider_1, slider_2, slider_3 = st.columns(3)
    # Create range sliders for filtering
    min_rent, max_rent = int(main_df["baseRent"].min()), int(main_df["baseRent"].max())
    rent_range = slider_1.slider('Base Rent Range', min_value=min_rent, max_value=max_rent,
                                 value=(min_rent, max_rent))

    min_year, max_year = int(main_df["yearConstructed"].min()), int(main_df["yearConstructed"].max())
    year_range = slider_2.slider('Year Constructed Range', min_value=min_year, max_value=max_year,
                                 value=(min_year, max_year))

    min_space, max_space = int(main_df["livingSpace"].min()), int(main_df["livingSpace"].max())
    space_range = slider_3.slider('Living Space Range (sqm)', min_value=min_space, max_value=max_space,
                                  value=(min_space, max_space))

    sort_column = slider_1.selectbox('Select column to sort by:', [
        "None",
        "baseRent",
        "totalRent"
        "yearConstructed",
        "livingSpace",
        "heatingCosts",
        "noRooms"
    ])


filtered_table = process.filter_table(main_df, rent_range, year_range, space_range, sort_column)

pagination = st.container()
bottom_menu = st.columns((4, 1, 1))
with bottom_menu[2]:
    batch_size = st.selectbox("Page Size", options=[25, 50, 100])
with bottom_menu[1]:
    total_pages = (
        int(len(filtered_table.head(100)) / batch_size)
        if int(len(filtered_table.head(100)) / batch_size) > 0
        else 1
    )
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
with bottom_menu[0]:
    st.markdown(f"Page **{current_page}** of **{total_pages}** ")


pages = split_frame(process.format_pandas_dataframe(filtered_table.head(100)), batch_size)
pagination.dataframe(data=pages[current_page - 1],
                     column_order=config.column_order,
                     use_container_width=True)

# ---- TABLE END ----------------

# Expanders : Creating expanders for each category ------

with st.expander("Location Details - Click for More Details"):
    for column, description in config.location_details.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Property Features - Click for More Details"):
    for column, description in config.property_features.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Amenities - Click for More Details"):
    for column, description in config.amenities.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Financial Details - Click for More Details"):
    for column, description in config.financial_details.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Construction and Efficiency - Click for More Details"):
    for column, description in config.construction_and_efficiency.items():
        st.markdown(f"- **{column}:** {description}")

with st.expander("Additional Descriptions - Click for More Details"):
    for column, description in config.additional_descriptions.items():
        st.markdown(f"- **{column}:** {description}")

# GRAPHS SECTION
st.markdown(
    "<h2 style='text-align: center;'>Explorative Data Analysis</h2><br>",
    unsafe_allow_html=True,
)

table_md = graphs.get_overview_info(main_df=main_df.to_pandas())
# Metric Table - Display the Markdown table in Streamlit
st.markdown(table_md, unsafe_allow_html=True)

# Calculating Metrics
median_rent, num_properties, avg_year_constructed = graphs.calculate_key_metrics(
    filtered_data
)

metric_cols = st.columns((2, 3, 3, 3))
with metric_cols[1]:

    st.metric(label="Median Rent (€)", value=f"{median_rent:.2f}")

with metric_cols[2]:
    st.metric(label="Number of Properties", value=num_properties)

with metric_cols[3]:
    st.metric(label="Avg. Year Constructed", value=f"{avg_year_constructed:.0f}")

# ------- Dynamic Map based on region -------------------------
st.markdown("<strong>Median Base Rent based on Region</strong>", unsafe_allow_html=True)
map_fig, final_df_pd = graphs.create_dynamic_map(df=filtered_data)
st.plotly_chart(map_fig, use_container_width=True)
insight_text_region_map = insights.generate_map_insights(final_df=final_df_pd)
st.markdown(insights.insight_container_style, unsafe_allow_html=True)
st.markdown(
    f"<div class='insight-container'>{insight_text_region_map}</div>",
    unsafe_allow_html=True,
)
# -------- TEXT GRAPHS (WORDCLOUD / TF-IDF-BARCHART )
text_graph_col1, text_graph_col2 = st.columns(2)
wordcloud_slider = text_graph_col1.slider("Max Number of Words", 50, 100, 250)
wordcloud_fig, word_barchart = graphs.generate_wordcloud(filtered_data, max_words=wordcloud_slider)

text_graph_col1.pyplot(wordcloud_fig, use_container_width=True)
text_graph_col2.plotly_chart(word_barchart, use_container_width=True)

# -----Distribution of baseRend and yearConstructed ------------
dist_col1, dist_col2 = st.columns(2)
dist_rent, dist_year = graphs.average_rent_year_distribution(df=main_df)
# histogram column 1
dist_col1.plotly_chart(dist_rent, use_container_width=True)
histogram_text_rental = insights.generate_histogram_insights(
    df=main_df, column_name="baseRent"
)
dist_col1.markdown(insights.insight_container_style, unsafe_allow_html=True)
dist_col1.markdown(
    f"<div class='insight-container'>{histogram_text_rental}</div>",
    unsafe_allow_html=True,
)
# histogram column 2
dist_col2.plotly_chart(dist_year, use_container_width=True)
histogram_text_year = insights.generate_histogram_insights(
    df=main_df, column_name="yearConstructed"
)
dist_col2.markdown(insights.insight_container_style, unsafe_allow_html=True)
dist_col2.markdown(
    f"<div class='insight-container'>{histogram_text_year}</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8" /> """,
    unsafe_allow_html=True,
)
col_1, col_2 = st.columns(2)

# Show the plot in Streamlit
with col_1:
    # 1. Graph - Rent Trend Analysis over Time
    rental_trend_fig, aggregated_df_average_rental = graphs.create_rent_trend_over_time(
        df=filtered_data
    )
    st.plotly_chart(rental_trend_fig, use_container_width=True)
    # Compute insights for rental trends in general
    insight_text_average = insights.extract_insights_average_rent_over_time(
        aggregated_df=aggregated_df_average_rental
    )
    col_1.markdown(insights.insight_container_style, unsafe_allow_html=True)
    col_1.markdown(
        f"<div class='insight-container'>{insight_text_average}</div>",
        unsafe_allow_html=True,
    )
    # ------------------------------------------------------------------------------------

with col_2:
    # Graph 2- Seasonal Rent Trend Chart
    rentals_per_season_fig, average_rent_by_season = (
        graphs.get_comparison_rent_by_season(df=filtered_data)
    )
    st.plotly_chart(rentals_per_season_fig, use_container_width=True)
    # Use markdown to display the insights in a stylized container
    insight_text_season = insights.extract_insights_average_rent_by_season(
        monthly_avg_rent_pd=average_rent_by_season
    )
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-container'>{insight_text_season}</div>",
        unsafe_allow_html=True,
    )
    # ---------------------------------------------------------------------------------


st.markdown(
    """<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8" /> """,
    unsafe_allow_html=True,
)
col_3, col_4 = st.columns(2)
with col_3:
    # rental by type bachart
    comparison_rent_by_prop = graphs.get_comparison_rent_by_property(df=filtered_data)
    # rental by type line chart
    rental_trend_by_type_fig, aggregated_df_average_rental_by_type = (
        graphs.create_rent_trend_over_time_by_type(df=filtered_data)
    )

    st.plotly_chart(comparison_rent_by_prop, use_container_width=True)

with col_4:

    # impact of prop on rental
    impact_prop_fig, impact_prop_df = graphs.get_impact_prop_features_on_rent(
        df=filtered_data
    )
    st.plotly_chart(impact_prop_fig, use_container_width=True)
    insight_impact_prop_text = insights.generate_dynamic_insights_impact_prop_on_rental(
        plot_data=impact_prop_df
    )
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-container'>{insight_impact_prop_text}</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    """<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8;" /> """,
    unsafe_allow_html=True,
)
col_5, col_6 = st.columns(2)
with col_5:

    # comparative analysis between old and new properties
    st.markdown(
        "<strong>Heatmap of Property Age vs. Rent Price Distribution</strong>",
        unsafe_allow_html=True,
    )
    age_property_fig, age_df = graphs.generate_age_rent_heatmap(df=filtered_data)
    st.plotly_chart(age_property_fig, use_container_width=True)

    age_text = insights.generate_age_heatmap_insights(pivot_df=age_df)
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-container'>{age_text}</div>",
        unsafe_allow_html=True,
    )
with col_6:

    # Impact Energy Efficiency
    energy_fig = graphs.plot_energy_efficiency_impact(df=filtered_data)
    st.plotly_chart(energy_fig, use_container_width=True)

st.markdown(
    """<hr style="height:1px;border:none;color:#E8E8E8;background-color:#E8E8E8;" /> """,
    unsafe_allow_html=True,
)

col_7, col_8 = st.columns(2)
with col_7:

    affordability_fig, avg_ratio_by_region_df = graphs.generate_affordability_graph(
        df=filtered_data
    )
    st.plotly_chart(affordability_fig, use_container_width=True)
    st.markdown(
        "<small>The average income data utilized in this analysis is derived from "
        "the <a href='https://en.wikipedia.org/wiki/List_of_German_states_by_household_income'>"
        "Wikipedia</a> page on household income in Germany, reflecting the "
        "household income per capita across the German states. </small>",
        unsafe_allow_html=True,
    )

    insight_affordability_text = insights.extract_insights_affordability(
        df=avg_ratio_by_region_df
    )
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-container'>{insight_affordability_text}</div>",
        unsafe_allow_html=True,
    )
with col_8:

    st.plotly_chart(rental_trend_by_type_fig, use_container_width=True)
    insight_text_by_prop = insights.extract_insights_average_rent_over_time_by_type(
        aggregated_df=aggregated_df_average_rental_by_type
    )
    st.markdown(insights.insight_container_style, unsafe_allow_html=True)
    st.markdown(
        f"<div class='insight-container'>{insight_text_by_prop}</div>",
        unsafe_allow_html=True,
    )
