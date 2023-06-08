import streamlit as st
import pandas as pd
import plotly_express as px
import pickle
import numpy as np

from PIL import Image

# configurations to modify name and icon
img = Image.open('icon.png')
st.set_page_config(page_title='Microsoft Engage 2022', page_icon=img)

# configurations to hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

page = ['EDA', 'Car Price Prediction', 'Other files Analysis']
choices = st.sidebar.selectbox('What would you like to see: ', page)

# declaring some global variables
global numeric_cols
global non_numeric_cols
global df
# dataframe importing
df = pd.read_csv('pre_processed_data.csv')

if choices == 'EDA':
    # dataframe filtering
    affordable = df[df["Ex-Showroom_Price"] <= 2000000]
    affordable.loc[:, 'Number_of_Airbags'] = affordable['Number_of_Airbags'].fillna(0)


    numeric_cols = list(affordable.select_dtypes(['float', 'int']).columns)
    non_numeric_cols = list(affordable.select_dtypes(['object']).columns)
    non_numeric_cols.append(None)

    # headings
    st.title('Welcome to my Microsoft Engage 2022 Data Analysis Project')
    st.subheader('Here I will be visualizing Automotive Dataset. ')

    # showing data in form of table
    st.write('cars_engage_2022')
    st.write('This data is taken from microsoft@acehacker website')

    # table
    st.subheader('Table cars_engage_2022')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    st.write(df)

    # pie chart
    st.subheader('Market share of different car manufacturers')
    fig1 = px.pie(df, values='Ex-Showroom_Price', names='Make')
    fig1.update_layout(width=550, margin=dict(l=1, r=1, b=1, t=1))
    st.write(fig1)

    st.text(" ")
    st.text(" ")

    # boxplot
    st.subheader('Boxplot below will tell the price range of different car manufacturers')
    fig2 = px.box(affordable, x="Make", y="Ex-Showroom_Price", color="Make")
    st.write(fig2)

    st.text(" ")
    st.text(" ")

    # scatter plot
    st.subheader('Make your own Scatter Plot by selecting the features ')
    x_input = st.selectbox('X axis', options=numeric_cols)
    y_input = st.selectbox('Y axis', options=numeric_cols)
    color_input = st.selectbox('Color', options=non_numeric_cols)
    fig3 = px.scatter(affordable, x=x_input, y=y_input,
                      color=color_input,
                      size='Ex-Showroom_Price',
                      hover_data=[x_input, color_input])
    st.write(fig3)

    st.text(" ")
    st.text(" ")

    # most common car specification combination
    # 5 number summary
    st.subheader('Most common car specification combination')
    st.write('Describing all the numerical columns of the dataset. The following table gives 5 number summary.')
    st.write(df.describe())

    st.text(" ")
    st.text(" ")

    # most repeated car specification
    st.write('This gives most common car specification of each feature.')
    st.write(df.mode(axis=0, numeric_only=False).head(2))

    st.text(" ")
    st.text(" ")

    # diving into safety
    st.header('Diving into safety Features')
    st.write('Start/Stop Button, Handbrake, Hill Assist, Child Safety Lock, Number of Airbags, Rain Sensing Wipers, Airbags, Parking Assistance are the features that play a major role in safety of people in the car. These features may help help in preventing accidents,life loses and minimizing car damage')
    input1 = st.selectbox('Select which feature would you like to analyze',
                       ['Start_/_Stop_Button', 'Handbrake', 'Hill_Assist', 'Child_Safety_Locks', "Number_of_Airbags",
                        'Rain_Sensing_Wipers', 'Parking_Assistance'])
    val_count = df[input1].value_counts()

    st.text(" ")
    st.text(" ")

    st.subheader('Pie Chart here shows the parts-to-whole relationship for safety feature selected')
    fig4 = px.pie(data_frame=df, values=val_count.values, names=val_count.index)
    st.plotly_chart(fig4)

    st.subheader('Analysis of Manufacturers v/s Ex-Showroom_Price with the safety feature selected')
    fig5 = px.bar(df, x='Make', y='Ex-Showroom_Price', color=input1)
    st.plotly_chart(fig5)

    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

if choices == 'Car Price Prediction':
    # import the model
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))

    st.title('Welcome to my Microsoft Engage 2022 Data Analysis Project')
    st.subheader('Car price predictor')

    # company
    make = st.selectbox('Company', df['Make'].unique())
    newdf = df[df["Make"] == make]

    # model
    Model = st.selectbox('Model', newdf['Model'].unique())

    c3, c4 = st.columns(2)

    with c3:
        # fuel type
        Fuel_Type = st.selectbox('Fuel Type', newdf['Fuel_Type'].unique())

    with c4:
        # Body type
        Body_Type = st.selectbox('Body Type', newdf['Body_Type'].unique())

    c5, c6 = st.columns(2)

    with c5:
        # ARAI certified mileage
        ARAI_Certified_Mileage = st.selectbox('ARAI certifies mileage', newdf['ARAI_Certified_Mileage'].unique())

    with c6:
        # Seating Capacity
        Seating_Capacity = st.selectbox('Seating Capacity', newdf['Seating_Capacity'].unique())

    c7, c8 = st.columns(2)

    with c7:
        # basic warranty
        Basic_Warranty = st.selectbox('Basic Warranty', newdf['Basic_Warranty'].unique())

    with c8:
        # Extended warranty
        Extended_Warranty = st.selectbox('Extended Warranty', newdf['Extended_Warranty'].unique())

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # start stop button
    Button = st.radio('Start/Stop button', newdf['Start_/_Stop_Button'].unique())

    # chile safety locks
    Child_Safety_Locks = st.radio('Child Safety Lock', newdf['Child_Safety_Locks'].unique())

    # number of airbags
    Number_of_Airbags = st.radio('Number_of_airbags', newdf['Number_of_Airbags'].unique())

    # parking assistance
    Parking_Assistance = st.radio('Parking Assistance', newdf['Parking_Assistance'].unique())

    # navigation system
    Navigation_System = st.radio('Navigation System', newdf['Navigation_System'].unique())

    if st.button('Predict Price'):
        # query
        query = np.array(
            [make, Model, Fuel_Type, Body_Type, ARAI_Certified_Mileage, Seating_Capacity, Button, Basic_Warranty,
             Child_Safety_Locks, Extended_Warranty, Number_of_Airbags, Parking_Assistance, Navigation_System])
        query = query.reshape(1, 13)
        st.title('The predicted price of car of such configuration is approximately Rs.' + str(int(np.exp(pipe.predict(query)[0]))))


if choices == 'Other files Analysis':
    # configurations
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # title of my app
    st.title('Welcome to my Microsoft Engage 2022 Data Analysis Project')

    # Sidebar
    # adding a sidebar
    st.sidebar.subheader("Here you can Visualize any dataset by yourself")

    # setting up file upload (here we will be able to upload file)
    uploaded_file = st.sidebar.file_uploader(label='Upload your CSV or Excel file.', type=['csv', 'xlsx'])

    # # declaring some global variables
    # global numeric_cols
    # global non_numeric_cols
    # global df

    # checking whether uploaded_file is csv or excel
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_csv(uploaded_file)

    try:
        st.write(df)
        numeric_cols = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_cols = list(df.select_dtypes(['object']).columns)
        non_numeric_cols.append(None)
    except Exception as e:
        print(e)
        st.write('Please upload file to the application.')

    # add a select widget to the sidebar
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplot', 'Histogram', 'Lineplot', 'Boxplot', 'Bargraph', 'Piechart']
    )

    # adding parameters for scatterplot
    if chart_select == 'Scatterplot':
        st.sidebar.subheader('Scatterplot Settings')
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_cols)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_cols)
            plot = px.scatter(data_frame=df, x=x_values, y=y_values)
            # display the chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    # adding parameters for Histogram
    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x_values = st.sidebar.selectbox('Feature', options=numeric_cols)
            bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_cols)
            plot = px.histogram(x=x_values, data_frame=df, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    # adding parameters for lineplot
    if chart_select == 'Lineplot':
        st.sidebar.subheader("Line Plot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_cols)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_cols)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_cols)
            plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    # adding parameters for Boxplot
    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            x_values = st.sidebar.selectbox("X axis", options=non_numeric_cols)
            y_values = st.sidebar.selectbox("Y axis", options=numeric_cols)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_cols)
            plot = px.box(data_frame=df, y=y_values, x=x_values, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    # adding parameters for bargraph
    if chart_select == 'Bargraph':
        st.sidebar.subheader('Bargraph Settings')
        try:
            x_values = st.sidebar.selectbox('X axis', options=non_numeric_cols)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_cols)
            # color_value = st.sidebar.selectbox("Color", options=non_numeric_cols)
            plot = px.bar(data_frame=df, y=y_values, x=x_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    # adding parameters for Piechart
    if chart_select == 'Piechart':
        st.sidebar.subheader('Piechart Settings')
        try:
            x_values = st.sidebar.selectbox('Names', options=non_numeric_cols)
            y_values = st.sidebar.selectbox('Values', options=numeric_cols)
            # color_value = st.sidebar.selectbox("Color", options=non_numeric_cols)
            plot = px.pie(data_frame=df, values=y_values, names=x_values,
                          color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(plot)

        except Exception as e:
            print(e)


st.write("")
st.write("")
st.write("")
st.write("")

footer = """<style>
a:link , a:visited{
color: white;
background-color: transparent;
}

a:hover,  a:active {
color: #e44646;
background-color: transparent;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
position:relative;
background-color: transparent;
color: white;
text-align: center;
}

</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://www.linkedin.com/in/himanshi-chainani" target="_blank">Himanshi Chainani.</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)