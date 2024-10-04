from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
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


st.title("Player Performance")


condensed_df = pd.read_csv('Streamlit/Rolling_SG_group_by_hole_player.csv')
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


st.success('''**A Brief Note on Methods:**  
I developed an expected strokes model to identify player performance, [check it out here!](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)''')
