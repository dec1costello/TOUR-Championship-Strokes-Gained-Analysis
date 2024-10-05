from bokeh.models import FixedTicker, ColumnDataSource, Whisker
from bokeh.transform import factor_cmap, jitter
from bokeh.plotting import figure, show
from bokeh.palettes import viridis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="TOUR Championship Player Performance Dashboard", 
    page_icon="⛳", 
    layout="centered",
    initial_sidebar_state="expanded")


st.sidebar.markdown(" ## About")
st.sidebar.markdown("This Dashboard offers deeper insights into a golfer's true abilities during the 2011 TOUR Championship. The primary aspiration is to contribute meaningful insights to the golf community."  )  
st.sidebar.markdown(" ## Resources")
st.sidebar.markdown("""- [Streamlit Documentation](https://docs.streamlit.io/)""")
st.sidebar.markdown(" ## Refrences")
st.sidebar.markdown("""- [Mark Broadie](https://www.amazon.com/Every-Shot-Counts-Revolutionary-Performance/dp/1592407501/ref=sr_1_1?crid=E9DM6O7HRG9D&dib=eyJ2IjoiMSJ9.ri2Cv9pHNJbS3-Hpa2UgOtOsruVYD1KjGoGxwiM2eA5_0USs190JqviRHOE0y2gmt8vgod8a7KUODYMy3gYlh1C5g2FivG0gWjLCIe_9Jbh8y-pGolXl46ApjwVSd1CGsxvVfx3h5x-WEgsGNMjcDg.yMWaO41yL01r9c2Q5ietMRsgZx6Sifafgw7ABpKN4Ts&dib_tag=se&keywords=strokes+gained+book&qid=1724565764&sprefix=strokes+gained+book%2Caps%2C127&sr=8-1)""")
st.sidebar.markdown(" ## Info")
st.sidebar.info("Read more about my code on my [Github](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis).", icon="ℹ️")
condensed_df = pd.read_csv('Streamlit/Rolling_SG_group_by_hole_player.csv')
df = pd.read_csv('Streamlit/player_profiles.csv')

st.title("TOUR Championship Analysis")
# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
#     st.image('https://github.com/dec1costello/Golf/assets/79241861/0f9673d0-36c6-4d6f-928b-34d171a19350')


profile_tab, Comparisons_tab, tab_faqs = st.tabs(["Profiles", "Comparisons", "FAQs"])


with profile_tab:
    col1, col2 = st.columns(2)
    with col1:
        golfer = st.selectbox("Select Player", condensed_df['last_name'].unique(), index=16)
        # Define the specific order for the 'SG_bins' categories
    df = df[df['last_name'] == golfer]
    order = ['OTT', '200+', '200-150', '150-100', '100-50', '50-0', 'Putting']
    
    # Convert the 'SG_bins' column to a categorical type with the specified order
    df['SG_bins'] = pd.Categorical(df['SG_bins'], categories=order, ordered=True)
    
    # Define the winter palette
    winter_palette = cm.get_cmap('winter', 8)
    
    # Map indices to colors
    def index_to_color(index):
        return tuple(int(c * 255) for c in winter_palette(index / 255)[:3])

    # Define indices for colors
    colors = {
        'OTT': index_to_color(0),
        '200+': index_to_color(40),
        '200-150': index_to_color(80),
        '150-100': index_to_color(120),
        '100-50': index_to_color(160),
        '50-0': index_to_color(200),
        'Putting': index_to_color(254)
    }
    
    categories = ['OTT', '200+', '200-150', '150-100', '100-50', '50-0', 'Putting']
    palette = [colors[cat] for cat in categories]
    
    # Create a mapping from SG_bins to numeric values
    sg_bins_mapping = {label: i for i, label in enumerate(categories)}
    df['SG_bins_numeric'] = df['SG_bins'].map(sg_bins_mapping)
    
    # Define the plot
    p2 = figure(height=400,width=700, title=f"{golfer}'s SG Percentile by Shot Type")
    p2.xaxis.major_label_text_font_size = '10pt' 
    p2.xaxis.axis_label = 'Shot Type'
    p2.yaxis.axis_label = 'SG Percentile'
    p2.xaxis.axis_label_text_font_size = '12pt'  # Increase x-axis label font size
    p2.yaxis.axis_label_text_font_size = '12pt'
    p2.title.text_font_size = '18pt'
    ###############################
    
    p2.xgrid.grid_line_color = None
    
    # Set the x-axis ticker
    years = sorted(df.SG_bins.unique())
    p2.xaxis.ticker = FixedTicker(ticks=list(sg_bins_mapping.values()))
    p2.xaxis.major_label_overrides = {i: label for label, i in sg_bins_mapping.items()}
    
    # Group by 'SG_bins' and calculate the quantiles
    g = df.groupby("SG_bins")
    upper = g['sg_binned_percentile'].quantile(0.75).reset_index()
    lower = g['sg_binned_percentile'].quantile(0.25).reset_index()
    
    # Merge upper and lower quantiles
    whisker_data = pd.merge(upper, lower, on="SG_bins", suffixes=('_upper', '_lower'))
    whisker_data['base'] = whisker_data['SG_bins'].map(sg_bins_mapping)
    
    # Create ColumnDataSource for whiskers
    source = ColumnDataSource(data=dict(base=whisker_data['base'],
                                        upper=whisker_data['sg_binned_percentile_upper'],
                                        lower=whisker_data['sg_binned_percentile_lower']))
    
    # Create Whiskers
    error = Whisker(base="base", 
                    upper="upper", 
                    lower="lower", 
                    source=source,
                    level="annotation",
                     line_width=2)
    error.upper_head.size = 20
    error.lower_head.size = 20
    p2.add_layout(error)
    
    # Add scatter plot
    p2.scatter(jitter("SG_bins_numeric", 
                      0.3, 
                      range=p2.x_range), 
                     "sg_binned_percentile", 
                     source=df, 
                     alpha=0.3, 
                     size=15, 
                     line_color="white",
                     color=factor_cmap("SG_bins", 
                                       palette=palette, 
                                       factors=categories))
    st.bokeh_chart(p2, use_container_width=True)



with Comparisons_tab:
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Golfer 1", condensed_df['last_name'].unique(), index=16)
    with col2:
        player2 = st.selectbox("Select Golfer 2", condensed_df['last_name'].unique(), index=18)
    desired_order = []
    desired_order.append(player1)
    desired_order.append(player2)
    
    
    condensed_df_filtered = condensed_df[(condensed_df['last_name'] == player1) | (condensed_df['last_name'] == player2)]
    condensed_df = condensed_df.sort_values(by=['player_id', 'round', 'hole'])
    # Calculate the rolling sum for each 'player_id'
    condensed_df['rolling_sum_sg_per_hole_per_round_per_player'] = condensed_df.groupby('player_id')['sg_per_hole_per_round_per_player'].rolling(window=72, min_periods=1).sum().reset_index(level=0, drop=True)
    # Create a new column 'round_hole_combination' with a numeric value representing the combination of 'round' and 'hole'
    condensed_df['round_hole_combination'] = (condensed_df['round'] - 1) * 18 + condensed_df['hole']
    # Now condensed_df contains a new column 'round_hole_combination' representing the combination of 'round' and 'hole' as a numeric value ranging from 1 to 72.
    original_df = condensed_df[['last_name', 'round_hole_combination', 'sg_per_hole_per_round_per_player', 'rolling_sum_sg_per_hole_per_round_per_player']]
    # Create a new DataFrame with additional rows
    extra_rows = pd.DataFrame(columns=original_df.columns)
    # Iterate through unique player_ids and add rows with specified conditions
    for player_id in original_df['last_name'].unique():
        extra_row = pd.DataFrame({'last_name': [player_id], 'round_hole_combination': [0], 'sg_per_hole_per_round_per_player': [0], 'rolling_sum_sg_per_hole_per_round_per_player': [0]})
        extra_rows = pd.concat([extra_rows, extra_row], ignore_index=True)
    new_df = pd.concat([original_df, extra_rows], ignore_index=True)
    new_df = new_df.sort_values(by='round_hole_combination')
    # Pivot the DataFrame
    pivot_df = new_df.pivot_table(index='round_hole_combination', columns='last_name', values='rolling_sum_sg_per_hole_per_round_per_player', fill_value=0)
    sorted_series = pivot_df.iloc[-1].sort_values(ascending=False)
    pivot_df = pivot_df[desired_order]
    
    
    # Convert the pivot table to a Bokeh ColumnDataSource
    source = ColumnDataSource(pivot_df)
    # Create a Bokeh figure
    p = figure(width=450, 
               height=450, 
               title='Rolling Sum of Strokes Gained',
               x_range=(0, 72),
               x_axis_label='Championship Hole', 
               y_axis_label='Strokes Gained')
    line_colors = viridis(3) #distinct color palette for lines
    for i, column in enumerate(desired_order):
        line_color = line_colors[i % len(line_colors)]
        #line_color = winter_palette[i]
        #line_color = winter_palette[i % len(winter_palette)]
        p.line(x='round_hole_combination', 
               y=column, 
               source=source,
               line_width=8, 
               line_alpha=0.6, 
               legend_label=column, 
               line_color=line_color)            
    
    
    # Customize the legend
    p.xaxis.axis_label_text_font_size = '18pt'  # Increase x-axis label font size
    p.yaxis.axis_label_text_font_size = '18pt'
    # p.legend.title = 'Player'
    p.title.text_font_size = '18pt'
    p.legend.label_text_font_size = '12pt'
    p.legend.location = "top_left" 
    p.legend.orientation = "vertical"  # Change the orientation to vertical
    p.legend.click_policy = "hide"
    st.bokeh_chart(p, use_container_width=True)


with tab_faqs:
            st.markdown("###🔎 Frequently Asked Questions")
            expand_faq3 = st.expander("What are Strokes Gained?")
            with expand_faq3:
                st.video("https://www.youtube.com/watch?v=MeNHbGhPFzU")
            expand_faq2 = st.expander("What machine learning model did you use and how was it trained?")
            with expand_faq2:    
                st.write('''I ensembled the best performing [lazy predict](https://lazypredict.readthedocs.io/en/latest/) models together using a [stack](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html). In this project, I leveraged [optuna's](https://optuna.org/#dashboard) CMAES Sampler to not only find the best parameters for each model in the stack resulting in minimized MAE, but also [data preprocessing scalers, encoders, imputation, and feature selection methods](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main/Creating%20Model/OptimizingUtils). All trails are fed with appropriate offline training data from a [feast](https://feast.dev/) feature store. I utilized an [mlflow](https://medium.com/infer-qwak/building-an-end-to-end-mlops-pipeline-with-open-source-tools-d8bacbf4184f) model registry to track all Optuna trials. Databricks is leveraged to store production ready models.''')
                st.image('https://github.com/user-attachments/assets/c4b0cbb0-290d-4a3a-8572-779a810cc1ed')
            expand_faq1 = st.expander("Where can I see the code for the model?")
            with expand_faq1:
                        st.write('''It's all on my [Github](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main)!''', unsafe_allow_html=True)


st.success('''**A Brief Note on Methods:**  
I developed an expected strokes model to identify player performance, [check it out here!](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)''')
