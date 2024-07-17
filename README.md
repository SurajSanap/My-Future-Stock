# My-Future-Stock

## Overview
This project aims to predict stock prices using machine learning techniques and provide actionable insights for investors and financial analysts. It utilizes historical stock data to build and evaluate predictive models, helping users make informed decisions in the stock market.

## Features
- **Data Collection**: Fetches historical stock data using Quandl API.
- **Data Preprocessing**: Cleans and preprocesses data to handle missing values and outliers.
- **Feature Engineering**: Extracts relevant features such as price differentials and technical indicators.
- **Model Development**: Implements machine learning models for price prediction and buy/sell recommendations.
- **Web Interface**: Integrates with Flask to provide a user-friendly web application for predictions.

## Project Structure
```
├── app.py                    # Flask application for web interface
├── models.py                 # Machine learning models and utilities
├── static/
│   ├── css/
│   │   └── style.css         # CSS styles for web interface
│   └── images/               # Logos and images used in the application
├── templates/
│   ├── about.html            # About page template
│   ├── contact.html          # Contact page template
│   └── index.html            # Main page template
├── README.md                 # This README file
├── requirements.txt          # Python dependencies
└── StockPricePred2024.ipynb       # Jupyter Notebook with project implementation
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/SurajSanap/My-Future-Stock.git
   cd My-Future-Stock
   ```
   
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
3. Run the Flask application:
   ```
   python app.py
   ```
   Access the application at `http://localhost:5000` in your web browser.

## Usage
- **Home**: View and interact with the main page for stock price predictions.
- **About**: Learn more about the project, including its objectives and benefits.
- **Contact**: Find contact information and links to GitHub and LinkedIn profiles.

## Contributing
Contributions are welcome! Fork the repository and create a pull request with your enhancements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits
- Developed by Suraj Sanap
- Data sourced from Quandl 
