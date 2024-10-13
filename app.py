import os
import google.generativeai as genai
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Class for model-related operations
class ChurnModel:
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, data):
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1]
        return predictions, probabilities

# Class for file handling and analysis
class FileHandler:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def save_file(self, file):
        filename = secure_filename(file.filename)
        filepath = os.path.join(self.upload_folder, filename)
        file.save(filepath)
        return filepath

    def delete_file(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

# Class for handling the chatbot with Gemini
class GeminiChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def ask_question(self, user_question):
        try:
            # Create the generative model instance
            gemini_model = genai.GenerativeModel('gemini-pro')
            prompt = f"Given the Telcom Customer Churn dataset, which contains information about telecom customers and their likelihood to churn, with 7043 rows (customers) and 21 columns (features), and the 'Churn' column being the target variable. The user is asking questions about the dataset, its features, and potential insights that can be derived from it. Please respond to the user's question by providing accurate and relevant information about the dataset. Your response should be clear, concise, and focused on the user's query. Thank you.{user_question}"
            response_gemini = gemini_model.generate_content(prompt)
            return response_gemini.text.replace('**', '').replace('*', '•')
        except Exception as e:
            return f"Error: {str(e)}"

# Class for visualization
class Visualizer:
    def __init__(self, df):
        self.df = df

    def generate_visualizations(self):
        html = ''
        fig = px.scatter(self.df, x='MonthlyCharges', y='TotalCharges')
        html += fig.to_html(include_plotlyjs='cdn')

        fig = self.create_kde_plot('MonthlyCharges', 'Monthly charges by churn')
        html += fig

        fig = self.create_kde_plot('TotalCharges', 'Total charges by churn')
        html += fig

        # Correlation heatmap
        corr = self.df.corr()
        fig = px.imshow(corr, text_auto=True)
        html += fig.to_html(include_plotlyjs='cdn')

        # Univariate plots
        html += self.uniplot('Partner_No', 'Distribution of Partner')
        html += self.uniplot('PaymentMethod_Bank transfer (automatic)', 'Distribution of PaymentMethod')
        html += self.uniplot('Contract_Month-to-month', 'Distribution of Contract')
        html += self.uniplot('TechSupport_No', 'Distribution of TechSupport')
        html += self.uniplot('SeniorCitizen', 'Distribution of SeniorCitizen')

        return html

    def create_kde_plot(self, column, title):
        Mth = sns.kdeplot(self.df[column][(self.df["Prediction"] == 0)], color="Red", shade=True)
        Mth = sns.kdeplot(self.df[column][(self.df["Prediction"] == 1)], ax=Mth, color="Blue", shade=True)
        Mth.legend(["No Churn", "Churn"], loc='upper right')
        Mth.set_ylabel('Density')
        Mth.set_xlabel(column)
        Mth.set_title(title)

        # Convert matplotlib figure to HTML
        canvas = FigureCanvas(Mth.get_figure())
        img = io.BytesIO()
        fig = Mth.get_figure()
        fig.savefig(img, format='png')
        img.seek(0)
        img_url = base64.b64encode(img.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_url}">'

    def uniplot(self, col, title):
        if col in self.df.columns:
            fig = px.bar(self.df, x=col, title=title)
            return fig.to_html(include_plotlyjs='cdn')
        return ''

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # More secure random key for Flask session management
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'ChurnAnalysisModel.sav'

# Create instances of classes
file_handler = FileHandler(UPLOAD_FOLDER)
churn_model = ChurnModel(MODEL_PATH)
chatbot = GeminiChatbot('GOOGLE_API_KEY')  # Replace with actual Google API key

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
            filepath = file_handler.save_file(file)
            try:
                data = pd.read_csv(filepath)
                predictions, probabilities = churn_model.predict(data)
                percentage_churn = (predictions == 1).mean() * 100
                predictions_details = [(i + 1, prediction, probability) for i, (prediction, probability) in enumerate(zip(predictions, probabilities))]

                if percentage_churn > 20:
                    warning = f"Warning: More than 20% ({percentage_churn:.2f}%) of the predictions indicate churn."

                output_data = data.copy()
                output_data['Prediction'] = predictions
                output_data['Probability'] = probabilities
                output_filepath = os.path.join(os.getcwd(), 'result.xlsx')
                output_data.to_excel(output_filepath, index=False)

            except Exception as e:
                warning = f"Failed to process the file for predictions: {str(e)}"
            finally:
                file_handler.delete_file(filepath)

    return render_template('analysis.html', predictions_details=predictions_details, warning=warning)

@app.route("/chat", methods=['GET', 'POST'])
def chat():
    gemini_response = None
    if request.method == 'POST':
        user_question = request.form['user_input']
        gemini_response = chatbot.ask_question(user_question)
    return render_template('chat.html', gemini_response=gemini_response)

@app.route("/faq", methods=['GET', 'POST'])
def faq():
    return render_template('faq.html')

@app.route("/vis", methods=['GET', 'POST'])
def vis():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filepath = file_handler.save_file(file)
            try:
                df = pd.read_excel(filepath)
                visualizer = Visualizer(df)
                visualizations = visualizer.generate_visualizations()
                return render_template('vis.html', visualizations=visualizations)
            except Exception as e:
                return f"Failed to process the file for visualizations: {str(e)}"
            finally:
                file_handler.delete_file(filepath)
    return render_template('vis.html')

if __name__ == "__main__":
    app.run(debug=True)
