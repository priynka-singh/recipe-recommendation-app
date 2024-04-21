# Flask Recipe Recommendation System
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg)](https://github.com/priynka-singh/recipe-recommendation-app/pull/new/master)

The Recipe Recommendation System provides users with personalized recipe suggestions based on input ingredients, cuisine preferences, diet type, and time constraints.
Visit [http://bit.ly/my-personal-recipes](http://bit.ly/my-personal-recipes) to access the deployed version of the Recipe Recommendation System hosted on Google Cloud Platform.

## Table of Contents
- [Overview](#overview)
- [Setting up the Environment](#setting-up-the-environment)
- [Usage](#usage)
- [Contributing](#contributing)
- [Project Structure](#project-structure)
- [License](#license)

## Overview
The Recipe Recommendation System is a Flask-based web application designed to help users discover new recipes tailored to their preferences. It utilizes an algorithm based on TF-IDF along with cosine similarity to analyze user input and generate personalized recipe suggestions. Users can specify ingredients, cuisine preferences, diet type, and time constraints to receive customized recommendations.

The system provides an intuitive interface for users to input their preferences and view recommended recipes. It also includes features for visualizing recipe ratings and nutrition information.

## Setting up the Environment

To run the Flask-based Recipe Recommendation System locally, you will need to set up a virtual environment and install the required dependencies. Complete the following steps:

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
    Open a web browser and navigate to [http://localhost:5000](http://localhost:5000) to view the Recipe Recommendation System.

By following these steps, you will have the Flask app up and running on your local machine.

## Usage
- **Input Ingredients:** Enter the ingredients you have on hand in the provided text area.
- **Select Preferences:** Choose your cuisine preferences, diet type, and desired cooking time from the dropdown menus.
- **Get Recommendations:** Click the "Get Recommendations" button to receive personalized recipe suggestions based on your input.
- **Explore Recipes:** Browse through the recommended recipes, view ingredients, cooking steps, user ratings, and nutrition information.

Once you have familiarized yourself with the application locally, you can also try it out live on the web. Visit [http://bit.ly/my-personal-recipes](http://bit.ly/my-personal-recipes) to access the deployed version of the Recipe Recommendation System hosted on Google Cloud Platform.


## Contributing
Contributions to the Recipe Recommendation System are welcome! Here is how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

Please ensure that your code adheres to the project's coding standards and includes appropriate documentation.

## Project Structure

- **templates**: Includes HTML templates used for rendering web pages.
- **static**: Stores static files such as CSS, JavaScript, and images.
- **app.yaml**: Configuration file for deploying the application on Google Cloud Platform.
- - **app.py**: Python script for running the flask app.
- **Dockerfile**: Defines the steps to create a Docker image for the Flask application.
- **notebooks**: Includes Python notebooks for data cleaning, processing and modelling
  
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
