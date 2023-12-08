import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Load the model
model = joblib.load(open("model-v1.joblib", "rb"))

# Data preprocessing function
def data_preprocessor(df):
    df.wine_type = df.wine_type.map({'white': 0, 'red': 1})
    return df

# Visualization function for confidence level
def visualize_confidence_level(prediction_proba):
    data = (prediction_proba[0] * 100).round(2)
    grad_percentage = pd.DataFrame(data=data, columns=['Percentage'], index=['Low', 'Ave', 'High'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()

# Streamlit app header
st.write("""
# Wine Quality Prediction ML Web-App 
This app predicts the ** Quality of Wine **  using **wine features** input via the **side panel** 
""")

# Display wine image using Streamlit
image = Image.open('wine_image.png')
st.image(image, caption='wine company', use_column_width=True)

# Improved User Input Display with Styled Headers and Data Cells
st.sidebar.title('User Input Parameters')

def get_user_input():
    wine_type = st.sidebar.selectbox("Select Wine type", ("white", "red"))
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 3.8, 15.9, 7.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.08, 1.58, 0.4)
    citric_acid  = st.sidebar.slider('Citric Acid', 0.0, 1.66, 0.3)
    residual_sugar  = st.sidebar.slider('Residual Sugar', 0.6, 65.8, 10.4)
    chlorides  = st.sidebar.slider('Chlorides', 0.009, 0.611, 0.211)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 289, 200)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 440, 150)
    density = st.sidebar.slider('Density', 0.98, 1.03, 1.0)
    pH = st.sidebar.slider('pH', 2.72, 4.01, 3.0)
    sulphates = st.sidebar.slider('Sulphates', 0.22, 2.0, 1.0)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 14.9, 13.4)
    
    features = {'Wine Type': wine_type,
                'Fixed Acidity': fixed_acidity,
                'Volatile Acidity': volatile_acidity,
                'Citric Acid': citric_acid,
                'Residual Sugar': residual_sugar,
                'Chlorides': chlorides,
                'Free Sulfur Dioxide': free_sulfur_dioxide,
                'Total Sulfur Dioxide': total_sulfur_dioxide,
                'Density': density,
                'pH': pH,
                'Sulphates': sulphates,
                'Alcohol': alcohol
            }
    data = pd.DataFrame(features, index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.sidebar.subheader('Current User Input:')
st.sidebar.markdown(
    """
    <style>
        .table th { background-color: #2c3e50; color: #ecf0f1; }
        .table td { background-color: #ecf0f1; }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.table(user_input_df.style.format({col: "{:.2f}" for col in user_input_df.columns}))

# Create sparklines using plotly
for col in user_input_df.columns:
    min_val = user_input_df[col].min()
    max_val = user_input_df[col].max()

    # Normalize values for the sparkline
    normalized_values = (user_input_df[col] - min_val) / (max_val - min_val)

    # Create a sparkline using plotly
    sparkline_fig = go.FigureWidget(
        go.Scatter(
            x=user_input_df.index,
            y=normalized_values,
            mode='lines+markers',
            line=dict(color='royalblue', width=2),
            marker=dict(color='red', size=6),
        )
    )

    # Update the layout for better visualization
    sparkline_fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=50,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, range=[0, 1]),
    )

    st.sidebar.write(f"**{col.capitalize()} Sparkline**")
    st.sidebar.plotly_chart(sparkline_fig)

st.subheader('User Input parameters:')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)
