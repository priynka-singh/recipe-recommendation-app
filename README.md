# Flask Recipe Recommendation System
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg)](https://github.com/priynka-singh/recipe-recommendation-app/pull/new/master)

A Recipe Recommendation System built with Flask, providing users with personalized recipe suggestions based on input ingredients, cuisine preferences, diet type, and time constraints.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Recipe Recommendation System is a Flask-based web application designed to help users discover new recipes tailored to their preferences. It utilizes an algorithm based on TF-IDF along with cosine similarity to analyze user input and generate personalized recipe suggestions. Users can specify ingredients, cuisine preferences, diet type, and time constraints to receive customized recommendations.

The system provides an intuitive interface for users to input their preferences and view recommended recipes. It also includes features for visualizing recipe ratings and nutrition information.

## Setting up the Environment

To run the Flask Recipe Recommendation System locally, you'll need to set up a virtual environment and install the required dependencies. Here's how you can do it:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/priynka-singh/recipe-recommendation-app.git
    cd recipe-recommendation-app
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source env/bin/activate
        ```

4. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the Flask application:**
    ```sh
    python app.py
    ```

6. **Access the application:**
    Open a web browser and go to [http://localhost:5000](http://localhost:5000) to view the Recipe Recommendation System.

By following these steps, you'll have the Flask app up and running on your local machine.

## Usage
- **Input Ingredients:** Enter the ingredients you have on hand in the provided text area.
- **Select Preferences:** Choose your cuisine preferences, diet type, and desired cooking time from the dropdown menus.
- **Get Recommendations:** Click the "Get Recommendations" button to receive personalized recipe suggestions based on your input.
- **Explore Recipes:** Browse through the recommended recipes, view ingredients, cooking steps, and additional details.

## Contributing
Contributions to the Recipe Recommendation System are welcome! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

Please ensure that your code adheres to the project's coding standards and includes appropriate documentation.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
