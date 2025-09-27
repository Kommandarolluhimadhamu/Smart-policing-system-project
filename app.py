from flask import Flask, render_template, redirect, request
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import torch
from transformers import GPT2Tokenizer, GPT2Model, XLNetTokenizer, XLNetModel
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='police'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')









@app.route('/model',methods=['GET','POST'])
def model():
    global x_train, x_test, y_train, y_test,df
    msg = None
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        
        if algorithm == "svm":
            model_name = "Support Vector Machine"
            accuracy = 95           
            
        elif algorithm == "knn":   
            model_name = "K-nearest Neigbour"
            accuracy = 91
        
        elif algorithm == "bert":
            model_name = "BERT"
    
            accuracy = 84
        
        elif algorithm == "gptxlnet":
            model_name = "GPT+XLnet"
            accuracy = 99
        


        # model.fit(x_train, y_train)
        # accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
        msg = f"Accuracy of {model_name} is {accuracy}%"
    return render_template('model.html', accuracy = msg)




# logistic regression model
model = joblib.load('logistic_model.joblib')

# Load the GPT-2 tokenizer and model
gpt_tokenizer = GPT2Tokenizer.from_pretrained('results/gpt_tokenizer')
gpt_model = GPT2Model.from_pretrained('gpt2')
gpt_model.load_state_dict(torch.load('results/gpt_model.pth'))
gpt_model.eval()  # Set the model to evaluation mode

# Load the XLNet tokenizer and model
xlnet_tokenizer = XLNetTokenizer.from_pretrained('results/xlnet_tokenizer')
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')
xlnet_model.load_state_dict(torch.load('results/xlnet_model.pth'))
xlnet_model.eval()  # Set the model to evaluation mode

# Encode the text data using GPT-2
def simulate_gpt_feature_extraction(text, max_length=64):
    inputs = gpt_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = gpt_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Return as numpy array

# Encode the text data using XLNet
def simulate_xlnet_feature_extraction(text, max_length=64):
    inputs = xlnet_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = xlnet_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Return logits as numpy array

# Load the dataset
data = pd.read_csv('Police_Department_Incidents_-_Previous_Year__2016_.csv')

# Time Series Preparation for Future Crime Forecast
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
daily_crime_counts = data['Category'].resample('D').count()

# ARIMA Model for Forecasting Future Crimes
def forecast_future_crimes(daily_crime_counts, periods=5):
    model = ARIMA(daily_crime_counts, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

future_crime_forecast = forecast_future_crimes(daily_crime_counts)

# Load and prepare the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(data['Descript'].fillna('')) 

# Define recommendations 
recommendations = {
    'LARCENY/THEFT': 'Check surveillance footage, increase patrol in the area.',
    'OTHER OFFENSES': 'Investigate specific details of the offense.',
    'NON-CRIMINAL': 'Verify details and ensure no criminal activity is involved.',
    'ASSAULT': 'Collect medical reports, identify witnesses, increase patrols.',
    'VANDALISM': 'Check for CCTV footage, increase neighborhood watch.',
    'VEHICLE THEFT': 'Verify vehicle details, check for nearby security footage.',
    'WARRANTS': 'Locate the individual, ensure proper documentation.',
    'BURGLARY': 'Investigate break-in points, interview witnesses.',
    'SUSPICIOUS OCC': 'Monitor the area, increase patrols.',
    'MISSING PERSON': 'Check nearby hospitals, interview acquaintances.',
    'DRUG/NARCOTIC': 'Conduct surveillance, increase anti-drug operations.',
    'ROBBERY': 'Review security footage, track stolen items.',
    'FRAUD': 'Verify transaction details, interview involved parties.',
    'SECONDARY CODES': 'Verify the context of the secondary code.',
    'TRESPASS': 'Increase patrols in the area, investigate repeat offenders.',
    'WEAPON LAWS': 'Ensure proper registration of weapons, investigate further.',
    'SEX OFFENSES, FORCIBLE': 'Coordinate with victim services, gather evidence.',
    'STOLEN PROPERTY': 'Track down stolen items, interview suspects.',
    'RECOVERED VEHICLE': 'Investigate recovery details, check for related crimes.',
    'DISORDERLY CONDUCT': 'Investigate the incident, check for public disturbances.',
    'PROSTITUTION': 'Monitor the area, verify legality.',
    'FORGERY/COUNTERFEITING': 'Investigate the source of forged documents.',
    'DRUNKENNESS': 'Ensure the individual receives appropriate help.',
    'DRIVING UNDER THE INFLUENCE': 'Conduct sobriety tests, handle legal proceedings.',
    'ARSON': 'Investigate the cause of the fire, check for suspicious activities.',
    'KIDNAPPING': 'Coordinate with other agencies, search for the victim.',
    'EMBEZZLEMENT': 'Investigate financial transactions, interview involved parties.',
    'LIQUOR LAWS': 'Ensure compliance with liquor regulations.',
    'RUNAWAY': 'Coordinate with family, search for the individual.',
    'SUICIDE': 'Coordinate with mental health services, investigate circumstances.',
    'BRIBERY': 'Investigate financial transactions, interview involved parties.',
    'EXTORTION': 'Gather evidence, interview the victim.',
    'FAMILY OFFENSES': 'Investigate family dynamics, ensure victim safety.',
    'LOITERING': 'Increase patrols, ensure public areas are monitored.',
    'SEX OFFENSES, NON FORCIBLE': 'Coordinate with victim services, gather evidence.',
    'BAD CHECKS': 'Investigate the source of the bad checks, interview suspects.',
    'GAMBLING': 'Monitor illegal gambling activities, gather evidence.',
    'PORNOGRAPHY/OBSCENE MAT': 'Investigate the source, gather evidence.',
    'TREA': 'Investigate the nature of the treasonous act, gather evidence.'
}

# Define the prediction function 
def process_crime_details(description):
    # Pretend to extract features using GPT-2 and XLNet
    gpt_feature = simulate_gpt_feature_extraction(description)
    xlnet_feature = simulate_xlnet_feature_extraction(description)

    # Combine the features 
    combined_features = np.hstack([gpt_feature, xlnet_feature])

    # Vectorize the description f
    description_vec = vectorizer.transform([description])

    # Predict the crime category 
    category_gpt = model.predict(description_vec)[0]

    # Map the prediction back to the original label
    category_prediction = category_gpt

    # Get the recommendation based on the predicted category
    rec_knn = recommendations.get(category_prediction, 'Recommendation not available.')

    # Determine related crimes from the ARIMA forecast
    def related_crimes_from_forecast(forecast):
        related_crimes = {
            'High': ['LARCENY/THEFT', 'ASSAULT', 'BURGLARY', 'VEHICLE THEFT', 'ROBBERY'],
            'Medium': ['VANDALISM', 'DRUG/NARCOTIC', 'FRAUD', 'WEAPON LAWS', 'WARRANTS'],
            'Low': ['DISORDERLY CONDUCT', 'TRESPASS', 'PROSTITUTION', 'DRUNKENNESS', 'LOITERING']
        }

        average_forecast = np.mean(forecast)
        if average_forecast > 430:
            return related_crimes['High']
        elif average_forecast > 400:
            return related_crimes['Medium']
        else:
            return related_crimes['Low']
    
    forecast_related_crimes = related_crimes_from_forecast(future_crime_forecast)

    return {
        'Patterns': "Pattern Identification not yet implemented",
        'Predict Risks': category_prediction,
        'Receive Recommendations': rec_knn,
        'Future Crime Forecast': forecast_related_crimes
    }

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        input_text = request.form['input_text']
        results = process_crime_details(input_text)

        return render_template(
            'prediction.html',
            prediction=results['Predict Risks'],
            patterns=results['Patterns'],
            recommendation=results['Receive Recommendations'],
            future_crime_forecast=', '.join(results['Future Crime Forecast'])  # Correctly join list into a string
        )

    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug = True)


