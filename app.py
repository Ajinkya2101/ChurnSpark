import os
import google.generativeai as genai
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
import plotly.express as px
from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'GOOGLE_API_KEY'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'ChurnAnalysisModel.sav'
model = pickle.load(open(MODEL_PATH, 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/analysis", methods=['GET', 'POST'])
def analysis():
    predictions_details = []
    warning = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(os.getcwd(), filename)
            file.save(filepath)

            try:
                data = pd.read_csv(filepath)
                predictions = model.predict(data)
                probabilities = model.predict_proba(data)[:, 1]
                percentage_churn = (predictions == 1).mean() * 100
                predictions_details = [(i + 1, prediction, probability) for i, (prediction, probability) in enumerate(zip(predictions, probabilities))]
                if percentage_churn > 20:
                    warning = f"Warning: More than 20% ({percentage_churn:.2f}%) of the predictions indicate churn."

                # Create a new DataFrame with the predictions and original features
                output_data = data.copy()
                output_data['Prediction'] = predictions
                output_data['Probability'] = probabilities

                # Save the output to a new Excel file named 'result.xlsx'
                output_filepath = os.path.join(os.getcwd(), 'result.xlsx')
                output_data.to_excel(output_filepath, index=False)

            except Exception as e:
                warning = f"Failed to process the file for predictions: {str(e)}"
            finally:
                os.remove(filepath)

    return render_template('analysis.html', predictions_details=predictions_details, warning=warning)

def ask_gemini(user_question):
    model = genai.GenerativeModel('gemini-pro')
    response_gemini = model.generate_content(f"Given the Telcom Customer Churn dataset, which contains information about telecom customers and their likelihood to churn, with 7043 rows (customers) and 21 columns (features), and the 'Churn' column being the target variable. The user is asking questions about the dataset, its features, and potential insights that can be derived from it. Please respond to the user's question by providing accurate and relevant information about the dataset. Your response should be clear, concise, and focused on the user's query. Thank you.{user_question}")
    response_gemini = response_gemini.text.replace('**', '').replace('*', '•')

    return response_gemini

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    gemini_response = None

    if request.method == 'POST':
        user_question = request.form['user_input']
        gemini_response = ask_gemini(user_question)

    return render_template('chat.html', gemini_response=gemini_response)

@app.route("/faq", methods=['GET', 'POST'])
def faq():
    return render_template('faq.html')

@app.route("/vis", methods=['GET', 'POST'])
def vis():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(os.getcwd(), filename)
            file.save(filepath)

            try:
                df = pd.read_excel(filepath)
                visualizations = generate_visualizations(df)
                return render_template('vis.html', visualizations=visualizations)
            except Exception as e:
                return f"Failed to process the file for visualizations: {str(e)}"
            finally:
                os.remove(filepath)
    return render_template('vis.html')

def generate_visualizations(df):
    html = ''
    
    # Scatter plot
    fig = px.scatter(df, x='MonthlyCharges', y='TotalCharges')
    html += fig.to_html(include_plotlyjs='cdn')
    
    # KDE plots
    Mth = sns.kdeplot(df.MonthlyCharges[(df["Prediction"] == 0)], color="Red", shade=True)
    Mth = sns.kdeplot(df.MonthlyCharges[(df["Prediction"] == 1)], ax=Mth, color="Blue", shade=True)
    Mth.legend(["No Churn", "Churn"], loc='upper right')
    Mth.set_ylabel('Density')
    Mth.set_xlabel('Monthly Charges')
    Mth.set_title('Monthly charges by churn')
    
    # Convert matplotlib figure to HTML
    canvas = FigureCanvas(Mth.get_figure())
    img = io.BytesIO()
    fig = Mth.get_figure()
    fig.savefig(img, format='png')
    img.seek(0)
    img_url = base64.b64encode(img.getvalue()).decode()
    html += f'<img src="data:image/png;base64,{img_url}">'
    
    Tot = sns.kdeplot(df.TotalCharges[(df["Prediction"] == 0)], color="Red", shade=True)
    Tot = sns.kdeplot(df.TotalCharges[( df["Prediction"] == 1)], ax=Tot, color="Blue", shade=True)
    Tot.legend(["No Churn", "Churn"], loc='upper right')
    Tot.set_ylabel('Density')
    Tot.set_xlabel('Total Charges')
    Tot.set_title('Total charges by churn')
    
    # Convert matplotlib figure to HTML
    canvas = FigureCanvas(Tot.get_figure())
    img = io.BytesIO()
    fig = Tot.get_figure()
    fig.savefig(img, format='png')
    img.seek(0)
    img_url = base64.b64encode(img.getvalue()).decode()
    html += f'<img src="data:image/png;base64,{img_url}">'
    
    # Correlation heatmap
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True)
    html += fig.to_html(include_plotlyjs='cdn')
    
    # Univariate plots
    def uniplot(df, col, title, hue=None):
        if col in df.columns:
            fig = px.bar(df, x=col, color=hue, title=title)
            return fig.to_html(include_plotlyjs='cdn')
        else:
            return ''
    
    html += uniplot(df, 'Partner_No', 'Distribution of Partner')
    html += uniplot(df, 'PaymentMethod_Bank transfer (automatic)', 'Distribution of PaymentMethod')
    html += uniplot(df, 'Contract_Month-to-month', 'Distribution of Contract')
    html += uniplot(df, 'TechSupport_No', 'Distribution of TechSupport')
    html += uniplot(df, 'SeniorCitizen', 'Distribution of SeniorCitizen')
    
    return html

if __name__ == "__main__":
    app.run(debug=True)