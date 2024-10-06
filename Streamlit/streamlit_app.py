from bokeh.models import FixedTicker, ColumnDataSource, Whisker, FactorRange
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

st.sidebar.image('https://github.com/dec1costello/Golf/assets/79241861/0f9673d0-36c6-4d6f-928b-34d171a19350')
st.sidebar.markdown(" ## ⛳ About 🏌️")
st.sidebar.markdown("This Dashboard offers deeper insights into a golfer's true abilities during the 2011 TOUR Championship. The primary aspiration is to contribute meaningful insights to the golf community."  )  
st.sidebar.info("Read more about my code on my [Github](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis).", icon="ℹ️")
condensed_df = pd.read_csv('Streamlit/Rolling_SG_group_by_hole_player.csv')
df = pd.read_csv('Streamlit/player_profiles.csv')


st.title("TOUR Championship Analysis")


profile_tab, Comparisons_tab, tab_faqs = st.tabs(["Profiles", "Comparisons", "FAQs"])


with profile_tab:

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        golfer = st.selectbox("Select Golfer", condensed_df['last_name'].unique(), index=21)
        st.balloons()
    df = df[df['last_name'] == golfer]

    st.header("⛳ Total SG by Shot Type")

    description_text_2 = """
     The plot directly below displays a players's 
    Total SG by shot type, providing a clear visualization of 
    his performance across different lies and distances.
    """
    description_2 = st.empty()
    description_2.write(description_text_2.format("all"))
    
    #--------------------------PLOT 2--------------------------------------------------------------------
    order = ['OTT', '200+', '200-150', '150-100', '100-50', '50-0', 'Putting']
    df['SG_bins'] = pd.Categorical(df['SG_bins'], categories=order, ordered=True)    
    winter_palette = cm.get_cmap('winter', 8)    
    def index_to_color(index):
        return tuple(int(c * 255) for c in winter_palette(index / 255)[:3])
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
    sg_bins_mapping = {label: i for i, label in enumerate(categories)}
    df['SG_bins_numeric'] = df['SG_bins'].map(sg_bins_mapping)    
    p2 = figure(height=400,width=700, title=f"{golfer}'s SG Percentile by Shot Type")
    p2.xaxis.major_label_text_font_size = '10pt' 
    p2.xaxis.axis_label = 'Shot Type'
    p2.yaxis.axis_label = 'SG Percentile'
    p2.xaxis.axis_label_text_font_size = '12pt'  # Increase x-axis label font size
    p2.yaxis.axis_label_text_font_size = '12pt'
    p2.title.text_font_size = '18pt'    
    p2.xgrid.grid_line_color = None    
    years = sorted(df.SG_bins.unique())
    p2.xaxis.ticker = FixedTicker(ticks=list(sg_bins_mapping.values()))
    p2.xaxis.major_label_overrides = {i: label for label, i in sg_bins_mapping.items()}    
    g = df.groupby("SG_bins")
    upper = g['sg_binned_percentile'].quantile(0.75).reset_index()
    lower = g['sg_binned_percentile'].quantile(0.25).reset_index()    
    whisker_data = pd.merge(upper, lower, on="SG_bins", suffixes=('_upper', '_lower'))
    whisker_data['base'] = whisker_data['SG_bins'].map(sg_bins_mapping)    
    source = ColumnDataSource(data=dict(base=whisker_data['base'],
                                        upper=whisker_data['sg_binned_percentile_upper'],
                                        lower=whisker_data['sg_binned_percentile_lower']))    
    error = Whisker(base="base", 
                    upper="upper", 
                    lower="lower", 
                    source=source,
                    level="annotation",
                     line_width=2)
    error.upper_head.size = 20
    error.lower_head.size = 20
    p2.add_layout(error)    
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

    #-----------------------PLOT 1----------------------

    testdf = df.groupby(['SG_bins','from_location_scorer'])['SG'].sum()
    testdf_df = testdf.to_frame().T
    
    data_dict_v2 = {'categories': [], 'counts': []}
    
    # Step 2: Process the DataFrame
    for sg_bin in testdf_df.columns.levels[0]:
        if sg_bin == 'OTT' or sg_bin == 'Putting':
            # Include 'OTT' and 'Putting' directly without summing
            for scorer in testdf_df[sg_bin].columns:
                data_dict_v2['categories'].append((sg_bin, scorer))
                data_dict_v2['counts'].append(testdf_df[sg_bin][scorer].sum())  # Sum over the rows for each column
        else:
            # Sum 'Fairway' and 'Other'
            fairway_sum = testdf_df[sg_bin]['Fairway'].sum()  # Sum over the rows for 'Fairway'
            other_sum = testdf_df[sg_bin].drop(columns='Fairway').sum().sum()  # Sum over the rows for other columns
            data_dict_v2['categories'].append((sg_bin, 'Fairway'))
            data_dict_v2['counts'].append(fairway_sum)
            data_dict_v2['categories'].append((sg_bin, 'Other'))
            data_dict_v2['counts'].append(other_sum)
    
    # Step 3: Convert the result dictionary to a DataFrame for verification
    result_df = pd.DataFrame(data_dict_v2)
    
    
    # Filter out zero values
    filtered_categories = []
    filtered_counts = []
    for category, count in zip(data_dict_v2['categories'], data_dict_v2['counts']):
        if count != 0:
            filtered_categories.append(category)
            filtered_counts.append(count)
    
    data_dict_v2_filtered = {'categories': filtered_categories, 'counts': filtered_counts}
    data_dict_v2 = data_dict_v2_filtered

    # Create a ColumnDataSource
    source = ColumnDataSource(data=data_dict_v2)
    categories = [x[1] for x in data_dict_v2['categories']]
    
    # Create the figure
    p3 = figure(x_range=FactorRange(*data_dict_v2['categories']),height=500,width=700, title=f"{golfer}'s SG by Shot Type",
               toolbar_location=None, tools="", output_backend="svg")
    custom_colors = ['#0000ff','#00e38e','#00aaaa','#00e38e','#00aaaa','#00e38e','#00aaaa','#00e38e','#00e38e', '#00aaaa','#00fe80','#00fe80'] 
    p3.vbar(x='categories', top='counts', width=0.9, source=source, line_color="white",alpha=0.8, 
           fill_color=factor_cmap('categories', palette=custom_colors, factors=categories, start=1, end=11))
    min_value = min(data_dict_v2['counts']) - 1
    max_value = max(data_dict_v2['counts']) + 1
    p3.y_range.start = min_value
    p3.y_range.end = max_value
    p3.x_range.range_padding = 0.1
    p3.xaxis.major_label_orientation = 1
    p3.xgrid.grid_line_color = None
    p3.xaxis.major_label_text_font_size = '10pt'
    p3.xaxis.axis_label = 'Shot Type'
    p3.yaxis.axis_label = 'SG'
    p3.xaxis.axis_label_text_font_size = '12pt'
    p3.yaxis.axis_label_text_font_size = '12pt'
    p3.title.text_font_size = '18pt'
    st.bokeh_chart(p3, use_container_width=True)

    st.header("⛳ SG Percentile by Shot Type")
    description_text_3 = """
    By looking at a players's SG Percentile, 
    we can see where he consistently underperformed in each shot type bin, 
    opposed to having one or two shots damage a bin.
    """
    description_3 = st.empty()
    description_3.write(description_text_3.format("all"))
    st.bokeh_chart(p2, use_container_width=True)


with Comparisons_tab:
    col1, col2,col3,col4 = st.columns(4)
    with col1:
        player1 = st.selectbox("Select Golfer 1", condensed_df['last_name'].unique(), index=16)
        st.balloons()
    with col2:
        player2 = st.selectbox("Select Golfer 2", condensed_df['last_name'].unique(), index=18)
        st.balloons()
    desired_order = []
    desired_order.append(player1)
    desired_order.append(player2)

    st.header("⛳ SG Progression")

    description_text_2 = """
    The plot directly below displays the golfers' 
    progression of Strokes Gained relative to the field
    throughout the 2011 TOUR Championship.
    """
    description_2 = st.empty()
    description_2.write(description_text_2.format("all"))
    
    
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
            st.markdown("### 🔎 Frequently Asked Questions")
            expand_faq3 = st.expander("What are Strokes Gained (SG)?")
            with expand_faq3:
                    description_text = """
                    By subtracting Expected Strokes (xS) from the result (R) of each shot 
                    we get a player's strokes gained or lost from that shot. 
                    """
                    description = st.empty()
                    description.write(description_text.format("all"))
                
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        st.latex(r'''
                        SG = xS - R
                        ''')
                    st.markdown("""
                    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/episode/3d226zD4cKGSqkFRwkH9nw?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
                    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/episode/0jKnJG9pSCNsNsOsoEwNrh?utm_source=generator" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
                    <br><br>
                    """, unsafe_allow_html=True)

                    st.video("https://www.youtube.com/watch?v=MeNHbGhPFzU")



    
            expand_faq2 = st.expander("What machine learning model did you use to predict Expect Strokes (xS) and how was the model trained?")
            with expand_faq2:    
                st.write('''I ensembled the best performing [lazy predict](https://lazypredict.readthedocs.io/en/latest/) models together using a [stack](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html). In this project, I leveraged [optuna's](https://optuna.org/#dashboard) CMAES Sampler to not only find the best parameters for each model in the stack resulting in minimized MAE, but also [data preprocessing scalers, encoders, imputation, and feature selection methods](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main/Creating%20Model/OptimizingUtils). All trials are fed with appropriate offline training data from a [feast](https://feast.dev/) feature store. I utilized an [mlflow](https://medium.com/infer-qwak/building-an-end-to-end-mlops-pipeline-with-open-source-tools-d8bacbf4184f) model registry to track all Optuna trials. Databricks is leveraged to store production ready models.''')
                st.image('https://github.com/user-attachments/assets/c4b0cbb0-290d-4a3a-8572-779a810cc1ed')
            expand_faq1 = st.expander("Where can I see the code for the model?")
            with expand_faq1:
                        st.write('''It's all on my [Github](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main)!''', unsafe_allow_html=True)


st.success('''**A Brief Note on Methods:**  
I developed an expected strokes model to identify player performance, 
[check it out here!](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)''')


