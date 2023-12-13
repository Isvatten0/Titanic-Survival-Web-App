import joblib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st

# Load the model
model = joblib.load(open('model.joblib','rb'))

# Data preprocessing function
def data_preprocessor(df):
    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
    df['Passenger_Class'] = df['Passenger_Class'].map({'1st': 1, '2nd': 2, '3rd' : 3})
    df['Embarked'] = df['Embarked'].map({'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2})
    df['Title'] = df['Title'].map({'Mr.': 0, 'Mrs.': 1, 'Miss.': 2, 'Master.': 3, 'Dr.': 4, 'Rev.': 5, 'Other': 6})
    return df

# Visualization function for confidence level
def visualize_confidence_level(prediction_proba):
    color_palette = sns.color_palette("Set2")

    data = (prediction_proba[0] * 100).round(2)
    grad_percentage = pd.DataFrame(data=data, columns=['Percentage'], index=['Succumbed', 'Survived'])

    fig, ax = plt.subplots(figsize=(8, 5))
    grad_percentage.plot(kind='barh', color=color_palette, ax=ax, zorder=10, width=0.6, edgecolor='black', alpha=0.8)

    ax.legend(["Confidence Level"], loc='upper right')
    ax.set_xlim(xmin=0, xmax=100)

    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_facecolor('#f0f0f0')

    ax.set_xlabel("Percentage (%) Confidence Level", labelpad=10, weight='bold', size=12)
    ax.set_ylabel("Survival", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level', fontdict={'fontsize': 16, 'fontweight': 'bold'}, loc='center')

    for index, value in enumerate(grad_percentage['Percentage']):
        ax.text(value + 2, index, f'{value:.2f}%', va='center', ha='left', color='black', fontsize=10)

    ax.axvline(x=50, linestyle='--', color='gray', alpha=0.8, linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

# Streamlit app header with custom CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #10478a;
    opacity: 0.9;
    background: linear-gradient(135deg, #1a6abf55 25%, transparent 25%) -7px 0/ 14px 14px, linear-gradient(225deg, #1a6abf 25%, transparent 25%) -7px 0/ 14px 14px, linear-gradient(315deg, #1a6abf55 25%, transparent 25%) 0px 0/ 14px 14px, linear-gradient(45deg, #1a6abf 25%, #10478a 25%) 0px 0/ 14px 14px;
    text-align: center; /* Center-align the text */
    color: #ffffff; 
    font-size: 18px;
}

h1 {
    white-space: nowrap;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #10478a;  /* Match the background color of the main content */
    color: #ffffff;  /* White text */
    padding: 20px;
    border-radius: 10px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
<style>
    .header-text {
        font-size: 48px;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
# <h1 class="header-text">Titanic Survival Prediction</h1>
# <p class="header-text">- Machine Learning Web App -</p>
# <p>This app predicts the <b>Survival of Titanic Passengers</b> using <b>Passenger Characteristics</b> input.</p>
# """, unsafe_allow_html=True)


# Display wine image using Streamlit with borders
image = Image.open('titanic-tragedy-remembrance-concept-free-vector.jpg')

# Add borders to the image
border_size = 10
border_color = 'white'
image_with_borders = Image.new('RGB', (image.width + 2 * border_size, image.height + 2 * border_size), border_color)
image_with_borders.paste(image, (border_size, border_size))

# Display the image with borders
st.image(image_with_borders, caption='cartoon image of the Titanic with borders', use_column_width=True)

# Improved User Input Display with Styled Headers and Data Cells
st.sidebar.title('User Input Parameters')

def get_user_input():
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    passenger_class = st.sidebar.selectbox("Passenger's Class", ('1st', '2nd', '3rd'))
    age = st.sidebar.slider('Age', 0, 100, 30)
    total_siblings_and_spouses_aboard = st.sidebar.slider('Total Siblings and Spouses Aboard', 0, 10, 0)
    total_parents_and_children_aboard = st.sidebar.slider('Total Parents and Children Aboard', 0, 10, 0)
    fare = st.sidebar.slider('Fare Paid', 0.0, 500.0, 50.0)
    embarked = st.sidebar.selectbox("Embarked From", ("Cherbourg", "Queenstown", "Southampton"))
    title = st.sidebar.selectbox("Passenger's Title", ("Mr.", "Mrs.", "Miss.", "Master.", 'Dr.', 'Rev.', "Other"))
    
    features = {
        'Passenger_Class': passenger_class,
        'Sex': sex,
        'Age': age,
        'total_siblings_and_spouses_aboard': total_siblings_and_spouses_aboard,
        'total_parents_and_children_aboard': total_parents_and_children_aboard,
        'Fare': fare,
        'Embarked': embarked,
        'Title': title
    }
    
    data = pd.DataFrame(features, index=[0])
    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)
prediction = model.predict(processed_user_input)
prediction_probability = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_probability)
# Prediction Outcome Statement
st.markdown("""
<style>
    .prediction-box {
        
        border-radius: 5px;
        text-align: center;
        font-size: 48px;
        margin-bottom: 40px;
        color: #FFFFFF;  /* White */
        background-color: transparent;
    }
    .survive-text {
        color: #FFD700;  /* Dark yellow */
        font-size: 36px;
    }
    .succumb-text {
        color: #FF0000;  /* Red */
        font-size: 36px;
    }
</style>
""", unsafe_allow_html=True)

if prediction == 1:
    st.markdown("""
    <div class="prediction-box">
        <p>This passenger is most likely to <b class="survive-text">Survive</b> on the Titanic.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="prediction-box">
        <p>This passenger is more likely to <b class="succumb-text">Succumb</b> to the Titanic.</p>
    </div>
    """, unsafe_allow_html=True)


# User Input parameters
st.write("""
# Definitions
""")

# Definitions
st.markdown("""
<style>
    .definitions {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        text-align: left;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="definitions">
    <b>Survived:</b> This variable indicates whether a passenger survived (1) or did not survive (0).
    <br>
    <b>Passenger Class (Pclass):</b> Represents the class of the ticket the passenger purchased (1st, 2nd, 3rd).
    <br>
    <b>Sex:</b> Gender of the passenger (Male or Female).
    <br>
    <b>Age:</b> Age of the passenger in years.
    <br>
    <b>Total Siblings and Spouses Aboard (SibSp):</b> Number of siblings or spouses the passenger had on the Titanic.
    <br>
    <b>Total Parents and Children Aboard (Parch):</b> Number of parents or children the passenger had on the Titanic.
    <br>
    <b>Fare:</b> Amount of money the passenger paid for the ticket.
    <br>
    <b>Embarked:</b> Port where the passenger boarded the Titanic.
    <br>
    <b>Title:</b> Used before passengers's name in order to show their status or profession.
</div>
""", unsafe_allow_html=True)

# How to Use Section
st.markdown("""
# How to Use
Welcome to the Titanic Survival Prediction app! This app helps you predict the likelihood of survival for Titanic passengers based on their characteristics. For optimal clarity, have your computer in dark mode.

### Input Parameters:
To make a prediction, follow these steps:
1. Select the passenger's characteristic using the dropdown menu / sliders on the left.
2. Click on the desired characteristic from the options.
3. Repeat with all desired changes to inputs.

### Prediction Confidence Level:
The bar chart displays the confidence level of the prediction. The x-axis represents the percentage, and the y-axis indicates whether the passenger is predicted to survive or not.

### Prediction Outcome:
After inputting the parameters, the app will predict whether the passenger is most likely to survive or succumb to the Titanic.

### Conclusion
You are now ready to make predictions! Adjust the input parameters, explore the visualizations, and discover the predicted outcomes for Titanic passengers.
""")



