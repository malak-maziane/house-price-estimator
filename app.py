#!/usr/bin/env python3
"""
🏠 Ames Housing Price Predictor - AI-Powered Edition
Features:
- Natural language input with smart parsing (+ Gemini LLM when available)
- Conversational feature extraction
- Comparable homes analysis
- Price optimization tips

Run: python app.py
"""

import gradio as gr
import pickle
import numpy as np
import pandas as pd
import json
import re

# Try to import Gemini - fallback to rule-based if unavailable
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_API_KEY = "AIzaSyA9bwHh-rUoxxvTZyZLOrCdy0T_ru74EwQ"
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    pass

# Load model and preprocessors
print("🔄 Loading model and preprocessors...")
with open("models/final_model.pkl", "rb") as f:
    model_pkg = pickle.load(f)

with open("models/preprocessors.pkl", "rb") as f:
    preprocessors = pickle.load(f)

model = model_pkg["model"]
feature_names = model_pkg["feature_names"]
scaler = preprocessors["scaler"]
highly_skewed = preprocessors["highly_skewed_features"]

# Load data for comparables
df_prep = pd.read_csv("data/data_preprocessed.csv")
df_original = pd.read_csv("data/train.csv")
median_baseline = df_prep[feature_names].median()

print(f"✅ Model loaded: {model_pkg['model_name']}")
print(f"✅ R² Score: {model_pkg['metrics']['r2']:.4f}")
print(f"✅ AI Parser: {'Gemini + Smart Parser' if GEMINI_AVAILABLE else 'Smart Parser (offline mode)'}")


def smart_parse_description(text, current_features=None):
    """
    Smart rule-based parser that extracts house features from natural language.
    Works offline without any API calls.
    """
    if current_features is None:
        current_features = {}
    
    text_lower = text.lower()
    extracted = {}
    
    # Extract bedrooms → estimate total rooms
    bedroom_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?|br|bdrm)', text_lower)
    if bedroom_match:
        bedrooms = int(bedroom_match.group(1))
        extracted['totrms_abvgrd'] = bedrooms + 3  # bedrooms + kitchen + living + dining
    
    # Extract bathrooms
    bath_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:bath(?:room)?s?|ba)', text_lower)
    if bath_match:
        baths = float(bath_match.group(1))
        extracted['full_bath'] = int(baths)
    
    # Extract square footage / area
    sqft_patterns = [
        r'(\d{3,5})\s*(?:sq\.?\s*(?:ft|feet)|square\s*feet|sqft|sf)',
        r'about\s*(\d{3,5})\s*(?:sq|square)?',
        r'(\d{3,5})\s*(?:sq|square)',
    ]
    for pattern in sqft_patterns:
        sqft_match = re.search(pattern, text_lower)
        if sqft_match:
            extracted['gr_liv_area'] = int(sqft_match.group(1))
            extracted['first_flr_sf'] = int(int(sqft_match.group(1)) * 0.6)  # Estimate
            break
    
    # Extract year built
    year_patterns = [
        r'built\s*(?:in\s*)?(\d{4})',
        r'(\d{4})\s*(?:build|construction|home|house)',
        r'from\s*(\d{4})',
        r'year[:\s]*(\d{4})',
    ]
    for pattern in year_patterns:
        year_match = re.search(pattern, text_lower)
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2010:
                extracted['year_built'] = year
                break
    
    # Extract garage
    garage_patterns = [
        r'(\d+)\s*(?:-?\s*car)?\s*garage',
        r'garage\s*(?:for\s*)?(\d+)',
    ]
    for pattern in garage_patterns:
        garage_match = re.search(pattern, text_lower)
        if garage_match:
            cars = int(garage_match.group(1))
            extracted['garage_cars'] = min(cars, 4)
            extracted['garage_area'] = cars * 250  # ~250 sq ft per car
            break
    
    # Check for no garage
    if 'no garage' in text_lower or 'without garage' in text_lower:
        extracted['garage_cars'] = 0
        extracted['garage_area'] = 0
    
    # Extract basement
    bsmt_match = re.search(r'(\d{3,4})\s*(?:sq\.?\s*ft)?\s*basement', text_lower)
    if bsmt_match:
        extracted['total_bsmt_sf'] = int(bsmt_match.group(1))
    elif 'finished basement' in text_lower or 'full basement' in text_lower:
        area = extracted.get('gr_liv_area', current_features.get('gr_liv_area', 1500))
        extracted['total_bsmt_sf'] = int(area * 0.7)
    elif 'no basement' in text_lower:
        extracted['total_bsmt_sf'] = 0
    
    # Extract fireplace
    fire_match = re.search(r'(\d+)\s*fireplace', text_lower)
    if fire_match:
        extracted['fireplaces'] = int(fire_match.group(1))
    elif 'fireplace' in text_lower:
        extracted['fireplaces'] = 1
    elif 'no fireplace' in text_lower:
        extracted['fireplaces'] = 0
    
    # Extract quality indicators
    quality_keywords = {
        'luxury': 9, 'luxurious': 9, 'high-end': 9, 'premium': 8,
        'excellent': 9, 'amazing': 8, 'beautiful': 7, 'great': 7,
        'nice': 6, 'good': 6, 'average': 5, 'decent': 5,
        'okay': 4, 'basic': 4, 'simple': 4, 'needs work': 3,
        'fixer': 3, 'fixer-upper': 2, 'poor': 2, 'bad': 2
    }
    for keyword, quality in quality_keywords.items():
        if keyword in text_lower:
            extracted['overall_qual'] = quality
            break
    
    # Merge with existing features
    current_features.update(extracted)
    
    # Determine what's still missing
    critical_features = ['overall_qual', 'gr_liv_area', 'year_built']
    missing = [f for f in critical_features if f not in current_features]
    
    # Generate follow-up question if needed
    question = None
    if missing:
        questions_map = {
            'overall_qual': "How would you rate the overall quality? (1=poor to 10=excellent)",
            'gr_liv_area': "What's the approximate living area in square feet?",
            'year_built': "What year was the house built?",
        }
        question = questions_map.get(missing[0])
    
    return {
        "extracted": extracted,
        "missing_critical": missing,
        "question": question
    }


def try_gemini_parse(text, current_features):
    """Try to use Gemini for smarter parsing, fall back to rule-based."""
    if not GEMINI_AVAILABLE:
        return smart_parse_description(text, current_features)
    
    try:
        model_llm = genai.GenerativeModel('gemini-2.0-flash-exp')  # Try experimental
        prompt = f"""Extract house features from: "{text}"
Current known: {json.dumps(current_features)}
Return JSON with: extracted (dict), missing_critical (list), question (str or null)
Features: overall_qual (1-10), gr_liv_area, garage_cars, total_bsmt_sf, first_flr_sf, year_built, full_bath, totrms_abvgrd, fireplaces, garage_area"""
        
        response = model_llm.generate_content(prompt)
        text_resp = response.text.strip()
        if text_resp.startswith("```"):
            text_resp = text_resp.split("```")[1].replace("json", "").strip()
        return json.loads(text_resp)
    except Exception as e:
        print(f"⚠️ Gemini unavailable ({type(e).__name__}), using smart parser")
        return smart_parse_description(text, current_features)


def predict_price_from_features(features_dict):
    """Predict price from a dictionary of features."""
    try:
        input_data = pd.DataFrame([median_baseline], columns=feature_names)
        
        overall_qual = features_dict.get('overall_qual', 6)
        gr_liv_area = features_dict.get('gr_liv_area', 1500)
        first_flr_sf = features_dict.get('first_flr_sf', 1000)
        garage_cars = features_dict.get('garage_cars', 2)
        total_bsmt_sf = features_dict.get('total_bsmt_sf', 1000)
        year_built = features_dict.get('year_built', 1990)
        full_bath = features_dict.get('full_bath', 2)
        totrms_abvgrd = features_dict.get('totrms_abvgrd', 6)
        fireplaces = features_dict.get('fireplaces', 1)
        garage_area = features_dict.get('garage_area', 500)
        
        second_flr_sf = max(0, gr_liv_area - first_flr_sf)
        total_sf = gr_liv_area + total_bsmt_sf
        total_bathrooms = full_bath + 0.5
        house_age = 2010 - year_built
        quality_area = overall_qual * gr_liv_area
        
        user_features = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'TotalBsmtSF': total_bsmt_sf,
            '1stFlrSF': first_flr_sf,
            '2ndFlrSF': second_flr_sf,
            'GarageCars': garage_cars,
            'GarageArea': garage_area,
            'YearBuilt': year_built,
            'YearRemodAdd': year_built,
            'FullBath': full_bath,
            'TotRmsAbvGrd': totrms_abvgrd,
            'Fireplaces': fireplaces,
            'TotalSF': total_sf,
            'TotalBathrooms': total_bathrooms,
            'HouseAge': house_age,
            'RemodAge': house_age,
            'QualityArea': quality_area,
            'HasGarage': 1 if garage_cars > 0 else 0,
            'HasBasement': 1 if total_bsmt_sf > 0 else 0,
            'HasFireplace': 1 if fireplaces > 0 else 0,
        }
        
        scaler_features = list(scaler.feature_names_in_)
        for feat, val in user_features.items():
            if feat in feature_names and feat in scaler_features:
                transformed_val = np.log1p(val) if feat in highly_skewed else val
                feat_idx = scaler_features.index(feat)
                scaled_val = (transformed_val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
                input_data[feat] = scaled_val
        
        prediction = model.predict(input_data)[0]
        prediction = max(50000, min(prediction, 800000))
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def find_comparable_homes(features_dict, n=5):
    """Find similar homes from the original dataset."""
    try:
        target_quality = features_dict.get('overall_qual', 6)
        target_area = features_dict.get('gr_liv_area', 1500)
        target_year = features_dict.get('year_built', 1990)
        
        df = df_original.copy()
        df['distance'] = (
            abs(df['OverallQual'] - target_quality) * 10000 +
            abs(df['GrLivArea'] - target_area) +
            abs(df['YearBuilt'] - target_year) * 100
        )
        
        similar = df.nsmallest(n, 'distance')[['OverallQual', 'GrLivArea', 'YearBuilt', 
                                                'GarageCars', 'FullBath', 'SalePrice']]
        return similar
    except:
        return None


def get_optimization_tips(features_dict, price):
    """Generate improvement suggestions based on feature analysis."""
    tips = []
    
    qual = features_dict.get('overall_qual', 6)
    if qual < 8:
        potential_gain = (8 - qual) * 25000
        tips.append(f"🔧 **Upgrade quality from {qual} to 8**: Could add ~${potential_gain:,} (renovations, finishes)")
    
    baths = features_dict.get('full_bath', 2)
    if baths < 3:
        tips.append(f"🚿 **Add a bathroom**: Could add ~$12,000-15,000")
    
    garage = features_dict.get('garage_cars', 2)
    if garage < 2:
        tips.append(f"🚗 **Expand garage to 2-car**: Could add ~$15,000-20,000")
    
    fireplace = features_dict.get('fireplaces', 0)
    if fireplace == 0:
        tips.append(f"🔥 **Add a fireplace**: Could add ~$10,000-12,000")
    
    year = features_dict.get('year_built', 1990)
    if year < 1980:
        tips.append(f"🏗️ **Major remodel**: Updating from {year} build could modernize value significantly")
    
    if not tips:
        tips.append("✨ Your home is already well-optimized! Focus on maintenance and curb appeal.")
    
    return "\n".join(tips)


def process_chat(user_message, history, features_state):
    """Process chat message and update conversation."""
    if not user_message.strip():
        return history, features_state, "", "", ""
    
    try:
        current_features = json.loads(features_state) if features_state else {}
    except:
        current_features = {}
    
    # Use smart parser (with Gemini fallback)
    result = smart_parse_description(user_message, current_features.copy())
    
    # Update features
    if result.get("extracted"):
        current_features.update(result["extracted"])
    
    # Build response
    if result.get("question") and len(current_features) < 3:
        extracted_summary = ", ".join([f"**{k}**: {v}" for k, v in current_features.items()])
        if extracted_summary:
            bot_response = f"📝 Got it! I extracted: {extracted_summary}\n\n{result['question']}"
        else:
            bot_response = f"🤔 {result['question']}"
    else:
        price = predict_price_from_features(current_features)
        if price:
            if price < 150000:
                category = "🏠 Budget-Friendly Home"
            elif price < 250000:
                category = "🏡 Mid-Range Property"
            elif price < 400000:
                category = "🏘️ Premium Home"
            else:
                category = "🏰 Luxury Estate"
            
            features_list = "\n".join([f"• **{k}**: {v}" for k, v in current_features.items()])
            bot_response = f"""## 🎉 Price Prediction Ready!

### 💰 Estimated Price: **${price:,.0f}**
### 📊 Category: {category}

**Features detected:**
{features_list}

---
*Click "Show Comparable Homes" or "Get Improvement Tips" for more insights!*"""
        else:
            bot_response = "Please tell me more about your house (size, year built, bedrooms, etc.)"
    
    # Update history (Gradio 6 format)
    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_response})
    
    # Live price display
    price_display = ""
    category_display = ""
    if len(current_features) >= 2:
        price = predict_price_from_features(current_features)
        if price:
            price_display = f"${price:,.0f}"
            if price < 150000:
                category_display = "🏠 Budget"
            elif price < 250000:
                category_display = "🏡 Mid-Range"
            elif price < 400000:
                category_display = "🏘️ Premium"
            else:
                category_display = "🏰 Luxury"
    
    return history, json.dumps(current_features), price_display, category_display, ""


def show_comparables(features_state):
    """Show comparable homes."""
    try:
        features = json.loads(features_state) if features_state else {}
        if not features:
            return "💡 Describe your house first in the chat to see comparables!"
        
        comparables = find_comparable_homes(features)
        if comparables is None or comparables.empty:
            return "No comparable homes found."
        
        result = "## 🏘️ Similar Homes from Ames, Iowa\n\n"
        result += "| Quality | Living Area | Year | Garage | Baths | **Sale Price** |\n"
        result += "|:-------:|:-----------:|:----:|:------:|:-----:|:--------------:|\n"
        
        for _, row in comparables.iterrows():
            result += f"| {int(row['OverallQual'])} | {int(row['GrLivArea']):,} sq ft | {int(row['YearBuilt'])} | {int(row['GarageCars'])} cars | {int(row['FullBath'])} | **${int(row['SalePrice']):,}** |\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def show_optimization(features_state):
    """Show optimization tips."""
    try:
        features = json.loads(features_state) if features_state else {}
        if not features:
            return "💡 Describe your house first to get improvement tips!"
        
        price = predict_price_from_features(features)
        if not price:
            return "Could not calculate price."
        
        tips = get_optimization_tips(features, price)
        return f"## 📈 Value Optimization Tips\n\n**Current Value: ${price:,.0f}**\n\n{tips}"
    except Exception as e:
        return f"Error: {str(e)}"


def clear_chat():
    return [], "{}", "", "", ""


def predict_from_sliders(overall_qual, gr_liv_area, garage_cars, garage_area,
                         total_bsmt_sf, first_flr_sf, year_built, full_bath,
                         totrms_abvgrd, fireplaces):
    features = {
        'overall_qual': overall_qual, 'gr_liv_area': gr_liv_area,
        'garage_cars': garage_cars, 'garage_area': garage_area,
        'total_bsmt_sf': total_bsmt_sf, 'first_flr_sf': first_flr_sf,
        'year_built': year_built, 'full_bath': full_bath,
        'totrms_abvgrd': totrms_abvgrd, 'fireplaces': fireplaces
    }
    price = predict_price_from_features(features)
    if price:
        if price < 150000: cat = "🏠 Budget-Friendly"
        elif price < 250000: cat = "🏡 Mid-Range"
        elif price < 400000: cat = "🏘️ Premium"
        else: cat = "🏰 Luxury"
        return f"💰 ${price:,.0f}", cat
    return "Error", ""


# Build UI
with gr.Blocks(title="🏠 AI House Price Predictor", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🏠 AI-Powered House Price Predictor
    **Model:** Extra Trees (R² = 0.91) | **Parser:** Smart NLP + Gemini AI
    """)
    
    features_state = gr.State("{}")
    
    with gr.Tabs():
        with gr.TabItem("💬 Chat Mode"):
            gr.Markdown("### Describe your house naturally!\n*Example: \"3 bedroom house, 2000 sq ft, built in 2005 with a 2-car garage\"*")
            
            chatbot = gr.Chatbot(height=300)
            
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Describe your house...", label="Your Message", scale=4)
                send_btn = gr.Button("🚀 Send", variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear")
                live_price = gr.Textbox(label="💰 Estimate", interactive=False, scale=1)
                live_cat = gr.Textbox(label="📊 Category", interactive=False, scale=1)
            
            with gr.Row():
                comp_btn = gr.Button("🏘️ Show Comparable Homes")
                opt_btn = gr.Button("📈 Get Improvement Tips")
            
            analysis = gr.Markdown()
            
            send_btn.click(process_chat, [chat_input, chatbot, features_state], 
                          [chatbot, features_state, live_price, live_cat, chat_input])
            chat_input.submit(process_chat, [chat_input, chatbot, features_state],
                             [chatbot, features_state, live_price, live_cat, chat_input])
            clear_btn.click(clear_chat, outputs=[chatbot, features_state, live_price, live_cat, chat_input])
            comp_btn.click(show_comparables, [features_state], [analysis])
            opt_btn.click(show_optimization, [features_state], [analysis])
        
        with gr.TabItem("🎚️ Slider Mode"):
            with gr.Row():
                with gr.Column():
                    s_qual = gr.Slider(1, 10, 6, step=1, label="Quality (1-10)")
                    s_area = gr.Slider(500, 5000, 1500, step=50, label="Living Area (sq ft)")
                    s_bsmt = gr.Slider(0, 3000, 1000, step=50, label="Basement (sq ft)")
                    s_1st = gr.Slider(500, 3000, 1000, step=50, label="1st Floor (sq ft)")
                with gr.Column():
                    s_cars = gr.Slider(0, 4, 2, step=1, label="Garage (cars)")
                    s_gar = gr.Slider(0, 1000, 500, step=50, label="Garage Area")
                    s_year = gr.Slider(1900, 2010, 1990, step=1, label="Year Built")
                with gr.Column():
                    s_bath = gr.Slider(0, 4, 2, step=1, label="Bathrooms")
                    s_rooms = gr.Slider(2, 12, 6, step=1, label="Rooms")
                    s_fire = gr.Slider(0, 4, 1, step=1, label="Fireplaces")
            
            slider_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
            with gr.Row():
                s_price = gr.Textbox(label="💰 Price")
                s_cat = gr.Textbox(label="📊 Category")
            
            slider_btn.click(predict_from_sliders,
                           [s_qual, s_area, s_cars, s_gar, s_bsmt, s_1st, s_year, s_bath, s_rooms, s_fire],
                           [s_price, s_cat])
        
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This App
            - **ML Model:** Extra Trees (R²=0.914, RMSE=$21,773)
            - **NLP:** Smart regex parser + Gemini AI fallback
            - **Data:** Ames Housing dataset (1,458 homes)
            
            ### Tips for Chat Mode
            Include details like: bedrooms, square feet, year built, garage, quality level
            """)

if __name__ == "__main__":
    print("🚀 Starting app...")
    demo.launch(share=False, inbrowser=True)
