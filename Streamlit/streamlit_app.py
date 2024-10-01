# from bokeh.plotting import figure
import streamlit as st
import pandas as pd
import pickle

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

tab_faq = st.tabs([ "FAQ"])

# with tab_ppredictor:
    
#     x = [1, 2, 3, 4, 5]
#     y = [6, 7, 2, 4, 5]
#     p = figure(title="simple line example", x_axis_label="x", y_axis_label="y")
#     p.line(x, y, legend_label="Trend", line_width=2)
#     st.bokeh_chart(p, use_container_width=True)

with tab_faq:
    
            st.markdown(" ### Frequently Asked Questions 🔎 ")

            expand_faq0 = st.expander("⛳ What dataset was used?")
            with expand_faq0:
                        st.write('''This dataset consists of shot level data from the PGA TOUR Championship. The TOUR Championship differs from other tournaments in that only the top 30 golfers compete and there's no cut after the second round, this ensures consistent data of high skill golfers across all 4 rounds. Additionally, it's important to acknowledge that the dataset lacks data from the playoff that occurred, which is crucial for understanding the tournament's conclusion. Furthermore, it is important to emphasize that landing in the rough at East Lake doesn't necessarily disadvantage a player. Despite the challenge it presents, the ball could still have a favorable lie, which might have been strategically chosen by the golfer.''', unsafe_allow_html=True)

            st.success('''**A Brief Note on Methods:**  

I developed an expected strokes model to identify player performance, [check it out here 😃!](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)''')
