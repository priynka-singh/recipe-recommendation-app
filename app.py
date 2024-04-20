import base64
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import json
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# Get environment variables 
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the processed recipe dataset
df = pd.read_csv("final_dataset.csv")

lemmatizer = WordNetLemmatizer()

# Preprocessing functions
def process_ingredients(ingredients_str):
    ingredients_str = ingredients_str.lower()
    l = ingredients_str.split()
    l = [lemmatizer.lemmatize(item) for item in l]
    return ' '.join(l)

# TF-IDF feature extractor
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['ingredients_combined'])

diet_type_dict = {
    'Gluten Free': 'diet_type_gluten_free',
    'Diary Free': 'diet_type_diary_free',
    'Special Nutrient Food': 'diet_type_special_nutrient_focus',
    'Low Carb': 'diet_type_low_carb',
    'Low Protein': 'diet_type_low_protein',
    'Diabetic Friendly': 'diet_type_diabetic_friendly',
    'Kosher': 'diet_type_kosher',
    'High Protein': 'diet_type_high_protein',
    'High Fiber': 'diet_type_high_fiber',
    'Vegetarian': 'diet_type_vegetarian',
    'Healthy': 'diet_type_healthy',
    'Low Calorie': 'diet_type_low_calorie',
    'Low Fat': 'diet_type_low_fat',
    'Low Saturated Fat': 'diet_type_low_saturated_fat',
    'Low Sodium': 'diet_type_low_sodium',
    'Low Cholesterol': 'diet_type_low_cholesterol',
    'Vegan': 'diet_type_vegan',
    'High Calcium': 'diet_type_high_calcium',
    'Lactose Free': 'diet_type_lactose_free',
    'Non Vegetarian': 'diet_type_non_vegetarian'}

cuisine_dict = {
    'Australian': 'cuisine_australian',
    'Canadian': 'cuisine_canadian',
    'European': 'cuisine_european',
    'Chinese': 'cuisine_chinese',
    'Middle Eastern': 'cuisine_middle_eastern',
    'American': 'cuisine_american',
    'African': 'cuisine_african',
    'Indian': 'cuisine_indian',
    'Russian': 'cuisine_russian',
    'Asian': 'cuisine_asian',
    'Oceanian': 'cuisine_oceanian',
    'Caribbean/Latin American': 'cuisine_caribbean_latin_american'}

# Get recipes function
def format_list_as_sentence(items):
    if not items:
        return "No data provided."
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ', and ' + items[-1] + '.'

def get_recipes_plot(input_str, time_to_make, diet_type, cuisine):
    if input_str != "":
        input_str = process_ingredients(input_str)
        input_tfidf = tfidf.transform([input_str])
        cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix)
        all_recipe_indices_sorted = cosine_similarities.argsort()[0][-30:][::-1]
        df_new = df.iloc[all_recipe_indices_sorted]

    elif input_str == "":
        df_new = df.sort_values(by=['num_rating', 'n_ingredients'], ascending=[False, True])

    if time_to_make != "No preference":
        df_new = df_new[df_new['time_to_make'] == time_to_make]

    if diet_type != ["No preference"]:
        for diet in diet_type:
            if diet in diet_type_dict:
                col_name = diet_type_dict[diet]
                df_new = df_new[df_new[col_name] == 1]

    if cuisine != "No preference":
        col_name2 = cuisine_dict[cuisine]
        df_new = df_new[df_new[col_name2] == 1]

    df_new = df_new[['name', 'recipe_id', 'minutes', 'n_steps', 'steps', 'ingredients',
                     'n_ingredients', 'cuisine', 'time_to_make', 'minutes_category',
                     'steps_category', 'calories', 'total_fat', 'sugar', 'sodium', 'protein',
                     'saturated_fat', 'carbohydrates', 'ingredients_combined', 'rating_1',
                     'rating_2', 'rating_3', 'rating_4', 'rating_5', 'num_rating']]

    top_recipes = df_new.sort_values(by=['n_ingredients', 'num_rating', 'minutes'], ascending=[True, False, True])

    if top_recipes.empty:
        return "Sorry! No recipes found, please try a different input!"
    
    top_recipes = top_recipes.head(10)
    # formatting
    top_recipes['name'] = top_recipes['name'].str.title()
    
    return top_recipes

def get_recipes_formatted(top_recipes):
    recipes_formatted = []
    for _, row in top_recipes.head(10).iterrows():
        recipe_dict = {
            "name": row['name'].title(),
            "ingredients": format_list_as_sentence(ast.literal_eval(row['ingredients'])),
            "steps": " ".join([step.capitalize() + '.' for step in ast.literal_eval(row['steps'])]),
        }
        recipes_formatted.append(recipe_dict)

    return recipes_formatted

import pandas as pd

import plotly.graph_objects as go

def plot_ratings(recipe_data):
    # Rename index (column) names
    recipe_data = recipe_data.rename(index={'rating_1': '1', 'rating_2': '2', 'rating_3': '3', 'rating_4': '4', 'rating_5': '5'})

    # Create a new dataframe for the current recipe
    ratings_df = pd.DataFrame({'Stars': ['1', '2', '3', '4', '5'],
                               'Frequency': recipe_data[['1', '2', '3', '4', '5']].values})
    
    # Pink shades in ascending order
    pink_shades = ['#EDCBD3', '#E1A6B4', '#D48195', '#C85D75', '#923147']

    # Create a new figure object
    fig = go.Figure()

    # Add a bar trace for the current recipe to the figure
    fig.add_trace(go.Bar(
        x=ratings_df['Stars'], 
        y=ratings_df['Frequency'],
        name=recipe_data['name'],  # Use recipe name as legend
        marker_color=pink_shades,  # Set colors for each bar
        width=0.9  # Make the bars touch each other
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Stars",
        yaxis_title="Frequency",
        template='plotly_white',  # Use light theme
        font_color='black',  # Set font color to black
        xaxis=dict(showgrid=False),  # Remove x-axis gridlines
        yaxis=dict(showgrid=False),  # Remove y-axis gridlines
        margin=dict(l=20, r=20, t=20, b=20),  # Set margins
        title=None,  # Remove the title
        width=400,  # Set width
        height=300   # Set height
    )
    return fig

import plotly.graph_objects as go

def plot_nutrition(data):
    '''
    Plots a horizontal bar chart showing the nutritional facts of each recipe
    Returns a plotly.graph_objects.Figure
    '''
    # Extracting the recipe name and capitalizing it
    recipe_title = f"Calories: {data['calories']}"

    # Extracting the nutrients and their values from the DataFrame
    nutrients = ['total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates']
    lilac_shades = ['#e2dcec', '#d4cce3', '#c7bcdb', '#b9add2', '#9f8dc1', '#846daf']  # Different shades of lilac for each nutrient

    y = [f"{nutrient.capitalize()}" for nutrient in nutrients]  # Nutrients without units
    x = [int(data[nutrient]) for nutrient in nutrients]  # Convert values to list

    fig = go.Figure()

    for nutrient, color, value in zip(nutrients, lilac_shades, x):
        # Capitalize the nutrient name
        if nutrient == 'total_fat':
            nutrient_name = 'Total fat'
        elif nutrient == 'saturated_fat':
            nutrient_name = 'Saturated fat'
        else:
            nutrient_name = nutrient.capitalize()

        fig.add_trace(go.Bar(
            x=[value],
            y=[f"{nutrient_name}"],  # Nutrient without units
            orientation='h',  # Horizontal bars
            marker_color=color,
            hovertemplate=f"<b>{nutrient_name}</b>: {value}",  # Add units as PDV to hover text
            text=value,  # Add units as PDV to labels
            textposition='outside',  # Place labels outside the bars
            textfont=dict(color='black'),  # Set label color to black
            showlegend=False,  # Hide legend
        ))

    fig.update_layout(
        template="plotly_white",  # Change the theme to white
        margin=dict(l=20, r=20, t=50, b=20),  # Increase top margin to 50
        width=500,
        height=250,  # Increase height to 250
        xaxis=dict(
            title='Value (PDV)',  # Add x-axis title with units
            showgrid=False,
        ),
        yaxis=dict(
            title='Nutrients',  # Add y-axis title
            showgrid=False,
        ),
        title=recipe_title,  # Set the title of the plot to the capitalized recipe title
        font=dict(color="black"),  # Set font color to black
    )

    return fig


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    data = request.json
    input_str = data.get('input_str', '')
    cuisine = data.get('cuisine', 'No preference')
    diet_type = data.get('diet_type', 'No preference')
    time_to_make = data.get('time_to_make', 'No preference')

    top_recipes = get_recipes_plot(input_str, time_to_make, diet_type, cuisine)

    response_data = []
    for index, recipe_data in top_recipes.head(10).iterrows():
        # Generate Plotly plots for each recipe
        ratings_plot = plot_ratings(recipe_data)
        nutrition_plot = plot_nutrition(recipe_data)

        # Convert Plotly figures to base64-encoded strings
        ratings_plot_b64 = ratings_plot.to_image(format="png")
        nutrition_plot_b64 = nutrition_plot.to_image(format="png")

        # Convert the images to base64 strings
        ratings_plot_b64_str = base64.b64encode(ratings_plot_b64).decode('utf-8')
        nutrition_plot_b64_str = base64.b64encode(nutrition_plot_b64).decode('utf-8')

        # Add the base64-encoded images to the recipe data
        recipe_data['ratings_plot'] = ratings_plot_b64_str
        recipe_data['nutrition_plot'] = nutrition_plot_b64_str

        # Convert DataFrame to dictionary and add to response_data
        recipe_dict = recipe_data.to_dict()
        response_data.append(recipe_dict)

    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True, port=5001)
