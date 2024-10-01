from bokeh.plotting import figure, show
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TOUR Championship Player Performance Dashboard", page_icon="⛳", initial_sidebar_state="expanded")

st.sidebar.markdown(" ## About")
st.sidebar.markdown("This Dashboard offers deeper insights into a golfer's true abilities. The primary aspiration is to contribute meaningful insights to the golf community."  )  


st.sidebar.markdown(" ## Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
""")

st.sidebar.markdown(" ## Refrences")
st.sidebar.markdown(
    """
- [Mark Broadie](https://www.amazon.com/Every-Shot-Counts-Revolutionary-Performance/dp/1592407501/ref=sr_1_1?crid=E9DM6O7HRG9D&dib=eyJ2IjoiMSJ9.ri2Cv9pHNJbS3-Hpa2UgOtOsruVYD1KjGoGxwiM2eA5_0USs190JqviRHOE0y2gmt8vgod8a7KUODYMy3gYlh1C5g2FivG0gWjLCIe_9Jbh8y-pGolXl46ApjwVSd1CGsxvVfx3h5x-WEgsGNMjcDg.yMWaO41yL01r9c2Q5ietMRsgZx6Sifafgw7ABpKN4Ts&dib_tag=se&keywords=strokes+gained+book&qid=1724565764&sprefix=strokes+gained+book%2Caps%2C127&sr=8-1)
""")

st.sidebar.markdown(" ## Info")
st.sidebar.info("Read more about my code on my [Github](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis).", icon="ℹ️")

st.title("Player Performance")

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]
p = figure(title="simple line example", x_axis_label="x", y_axis_label="y")
p.line(x, y, legend_label="Trend", line_width=2)
st.bokeh_chart(p, use_container_width=True)

df = pd.read_csv('https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/blob/main/Streamlit/Rolling_SG_group_by_hole_player.csv')
st.dataframe(df) 




st.success('''**A Brief Note on Methods:**  
I developed an expected strokes model to identify player performance, [check it out here!](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)''')
