"""
data/loader.py
--------------
Loads and preprocesses the Kaggle Food Recipe Dataset.
Caches processed data using Streamlit's cache for performance.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from nlp.preprocessor import clean_ingredients, preprocess_for_search

DATASET_PATH = "data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
SAMPLE_DATA_PATH = "data/sample_recipes.csv"


def create_sample_dataset() -> pd.DataFrame:
    """
    Creates a rich sample dataset when Kaggle CSV is not present.
    Covers all major cuisine types and ingredient categories.
    """
    recipes = [
        {
            "Title": "Classic Margherita Pizza",
            "Ingredients": "2 cups all-purpose flour, 1 tsp yeast, 1 cup tomato sauce, 2 cups mozzarella cheese, fresh basil leaves, olive oil, salt",
            "Instructions": "Mix flour and yeast to make dough. Let rise for 1 hour. Spread tomato sauce, top with mozzarella. Bake at 450°F for 15 minutes. Add fresh basil before serving.",
            "Image_Name": "pizza"
        },
        {
            "Title": "Chicken Tikka Masala",
            "Ingredients": "500g chicken breast, 1 cup yogurt, 2 tbsp garam masala, 1 tsp turmeric, 1 can tomatoes, 1 cup cream, onion, garlic, ginger",
            "Instructions": "Marinate chicken in yogurt and spices. Grill until charred. Make sauce with tomatoes, cream, and spices. Simmer chicken in sauce for 20 minutes.",
            "Image_Name": "chicken_tikka"
        },
        {
            "Title": "Avocado Toast with Poached Egg",
            "Ingredients": "2 slices sourdough bread, 1 ripe avocado, 2 eggs, lemon juice, red pepper flakes, salt, pepper, olive oil",
            "Instructions": "Toast bread. Mash avocado with lemon, salt, pepper. Poach eggs in simmering water for 3 minutes. Top toast with avocado and egg. Sprinkle red pepper flakes.",
            "Image_Name": "avocado_toast"
        },
        {
            "Title": "Mushroom Risotto",
            "Ingredients": "2 cups arborio rice, 500g mushrooms, 1 cup white wine, 4 cups vegetable broth, parmesan cheese, onion, garlic, butter, thyme",
            "Instructions": "Sauté mushrooms and set aside. Cook onion and garlic in butter. Add rice and toast. Add wine, then broth ladle by ladle, stirring. Finish with parmesan and mushrooms.",
            "Image_Name": "mushroom_risotto"
        },
        {
            "Title": "Chocolate Lava Cake",
            "Ingredients": "200g dark chocolate, 100g butter, 4 eggs, 4 egg yolks, 100g sugar, 50g flour, vanilla extract",
            "Instructions": "Melt chocolate and butter. Whisk eggs and sugar until pale. Fold in chocolate mixture and flour. Pour into greased ramekins. Bake at 200°C for 12 minutes.",
            "Image_Name": "lava_cake"
        },
        {
            "Title": "Caesar Salad",
            "Ingredients": "1 head romaine lettuce, 1/2 cup parmesan, croutons, 2 tbsp lemon juice, 1 tsp worcestershire sauce, 1 clove garlic, anchovies, olive oil, egg yolk",
            "Instructions": "Make dressing by blending garlic, anchovies, egg yolk, lemon, worcestershire, and olive oil. Toss lettuce with dressing, parmesan, and croutons.",
            "Image_Name": "caesar_salad"
        },
        {
            "Title": "Beef Tacos",
            "Ingredients": "500g ground beef, taco shells, 1 cup cheddar cheese, tomatoes, lettuce, sour cream, 2 tbsp taco seasoning, onion, lime",
            "Instructions": "Brown beef with taco seasoning. Warm taco shells. Fill with beef, cheese, tomatoes, lettuce. Top with sour cream and squeeze of lime.",
            "Image_Name": "beef_tacos"
        },
        {
            "Title": "Spaghetti Carbonara",
            "Ingredients": "400g spaghetti, 200g pancetta, 4 egg yolks, 1 cup pecorino romano, black pepper, salt, garlic",
            "Instructions": "Cook pasta al dente. Fry pancetta until crispy. Mix egg yolks with cheese and pepper. Toss hot pasta with pancetta, remove from heat, add egg mixture quickly.",
            "Image_Name": "carbonara"
        },
        {
            "Title": "Greek Salad",
            "Ingredients": "cucumber, tomatoes, red onion, kalamata olives, feta cheese, olive oil, oregano, lemon juice, salt",
            "Instructions": "Chop cucumber, tomatoes, and onion. Add olives and crumbled feta. Drizzle with olive oil and lemon. Season with oregano and salt.",
            "Image_Name": "greek_salad"
        },
        {
            "Title": "Butter Chicken",
            "Ingredients": "600g chicken, 1 cup tomato puree, 1/2 cup butter, 1 cup heavy cream, 2 tsp garam masala, 1 tsp cumin, ginger-garlic paste, kasuri methi",
            "Instructions": "Cook marinated chicken in butter. Add tomato puree and spices. Simmer for 20 min. Stir in cream and kasuri methi. Serve with naan.",
            "Image_Name": "butter_chicken"
        },
        {
            "Title": "Banana Pancakes",
            "Ingredients": "2 ripe bananas, 2 eggs, 1 cup flour, 1 cup milk, 1 tsp baking powder, maple syrup, butter, vanilla",
            "Instructions": "Mash bananas. Mix with eggs, flour, milk, baking powder. Cook on buttered pan until bubbles form. Flip. Serve with maple syrup.",
            "Image_Name": "pancakes"
        },
        {
            "Title": "Tomato Soup",
            "Ingredients": "1kg tomatoes, 1 onion, 3 garlic cloves, vegetable broth, 1 cup cream, basil, olive oil, salt, pepper, sugar",
            "Instructions": "Roast tomatoes with onion and garlic. Blend with broth. Strain and heat. Add cream. Season with salt, pepper, and sugar. Garnish with basil.",
            "Image_Name": "tomato_soup"
        },
        {
            "Title": "Pad Thai",
            "Ingredients": "200g rice noodles, 200g shrimp, 2 eggs, bean sprouts, green onions, 3 tbsp fish sauce, 2 tbsp tamarind paste, peanuts, lime",
            "Instructions": "Soak noodles. Stir fry shrimp in oil. Push aside, scramble eggs. Add noodles, sauce. Toss with bean sprouts. Top with peanuts and lime.",
            "Image_Name": "pad_thai"
        },
        {
            "Title": "Blueberry Muffins",
            "Ingredients": "2 cups flour, 1 cup blueberries, 3/4 cup sugar, 1/2 cup butter, 2 eggs, 1/2 cup milk, 1 tsp vanilla, 2 tsp baking powder",
            "Instructions": "Cream butter and sugar. Add eggs and vanilla. Mix in flour and baking powder alternating with milk. Fold in blueberries. Bake at 375°F for 20-25 minutes.",
            "Image_Name": "blueberry_muffins"
        },
        {
            "Title": "Garlic Shrimp Pasta",
            "Ingredients": "300g linguine, 400g large shrimp, 6 garlic cloves, 1/2 cup white wine, butter, parsley, lemon, chili flakes, salt",
            "Instructions": "Cook pasta. Sauté garlic in butter. Add shrimp and cook 2 min each side. Deglaze with wine. Toss with pasta, parsley, and lemon juice.",
            "Image_Name": "shrimp_pasta"
        },
        {
            "Title": "Vegetable Stir Fry",
            "Ingredients": "broccoli, bell peppers, snap peas, carrots, soy sauce, sesame oil, garlic, ginger, cornstarch, vegetable oil",
            "Instructions": "Heat oil in wok. Add garlic and ginger. Stir fry vegetables starting with hardest. Mix soy sauce with cornstarch. Pour over vegetables and toss.",
            "Image_Name": "stir_fry"
        },
        {
            "Title": "French Onion Soup",
            "Ingredients": "6 onions, 2 tbsp butter, 1 cup white wine, 4 cups beef broth, baguette slices, gruyere cheese, thyme, bay leaf, salt",
            "Instructions": "Caramelize onions in butter for 45 minutes. Add wine and reduce. Add broth and simmer. Ladle into oven-safe bowls. Top with bread and cheese. Broil until golden.",
            "Image_Name": "french_onion_soup"
        },
        {
            "Title": "Mango Smoothie",
            "Ingredients": "2 mangoes, 1 cup yogurt, 1/2 cup milk, 2 tbsp honey, lime juice, ice cubes, mint",
            "Instructions": "Peel and cube mangoes. Blend with yogurt, milk, honey, and ice. Add lime juice. Blend until smooth. Serve garnished with mint.",
            "Image_Name": "mango_smoothie"
        },
        {
            "Title": "Egg Fried Rice",
            "Ingredients": "3 cups cooked rice, 3 eggs, peas, carrots, soy sauce, sesame oil, green onions, garlic, vegetable oil",
            "Instructions": "Heat oil in wok. Scramble eggs, set aside. Fry garlic, add vegetables. Add cold rice and stir fry. Add eggs back, soy sauce, and sesame oil.",
            "Image_Name": "egg_fried_rice"
        },
        {
            "Title": "Lemon Cheesecake",
            "Ingredients": "500g cream cheese, 1 cup sugar, 3 eggs, 2 lemons, 1 cup graham crackers, 1/4 cup butter, 1 tsp vanilla, sour cream",
            "Instructions": "Make crust with crushed crackers and butter. Beat cream cheese and sugar. Add eggs, vanilla, lemon zest and juice. Pour over crust. Bake at 325°F for 50 minutes.",
            "Image_Name": "cheesecake"
        },
    ]
    df = pd.DataFrame(recipes)
    return df


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """
    Load dataset from CSV (Kaggle or sample).
    Returns a cleaned DataFrame ready for NLP processing.
    """
    df = None

    # Try loading the real Kaggle dataset
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            df = df.rename(columns={
                "Cleaned_Ingredients": "Ingredients",
                "title": "Title",
                "Title": "Title",
                "instructions": "Instructions",
                "Instructions": "Instructions",
            })
        except Exception:
            df = None

    # Fallback to sample dataset
    if df is None:
        df = create_sample_dataset()

    # ── Standardize columns ──
    df.columns = df.columns.str.strip()

    # Multiple source columns can normalize to the same canonical name after
    # renaming. Collapse duplicates so selecting a column always returns a Series.
    if df.columns.duplicated().any():
        deduped = {}
        for col in df.columns.unique():
            same_name = df.loc[:, df.columns == col]
            if same_name.shape[1] == 1:
                deduped[col] = same_name.iloc[:, 0]
            else:
                deduped[col] = same_name.bfill(axis=1).iloc[:, 0]
        df = pd.DataFrame(deduped)

    # Ensure required columns exist
    for col in ["Title", "Ingredients", "Instructions"]:
        if col not in df.columns:
            df[col] = ""

    df = df.dropna(subset=["Title"])
    df = df.reset_index(drop=True)

    # Clean ingredients
    df["Ingredients"] = df["Ingredients"].apply(
        lambda x: clean_ingredients(str(x)) if pd.notna(x) else ""
    )

    # Create search corpus (combined text for TF-IDF)
    df["search_corpus"] = df.apply(
        lambda row: preprocess_for_search(
            f"{row.get('Title', '')} {row.get('Ingredients', '')} {row.get('Instructions', '')}"
        ),
        axis=1
    )

    return df


def get_recipe_by_index(df: pd.DataFrame, idx: int) -> dict:
    """Return a single recipe dict by index."""
    if idx < 0 or idx >= len(df):
        return {}
    row = df.iloc[idx]
    return {
        "title": row.get("Title", "Unknown"),
        "ingredients": row.get("Ingredients", ""),
        "instructions": row.get("Instructions", ""),
        "image": row.get("Image_Name", ""),
    }
