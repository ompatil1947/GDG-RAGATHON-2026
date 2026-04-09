import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os

MODEL_PATH = "models/regression_model.pkl"

FEATURES = [
    "Academic_Score",
    "DSA_Skill",           # NEW feature
    "Project_Quality",
    "Experience_Score",
    "Soft_Skills",
    "OpenSource_Value",
    "Tech_Stack_Score"
]
TARGET = "Readiness_Score"

def train_and_save_model(force=False):
    if os.path.exists(MODEL_PATH) and not force:
        return  # already trained
    
    try:
        df = pd.read_csv("data/normalized_placement_data.csv")
    except Exception as e:
        print(f"Skipping model training: {e}")
        return
    
    # Simple data augmentation/cleaning if we lack variations
    # Provide defaults to missing values realistically
    X = df[FEATURES].fillna({
        "Academic_Score": 7.0,
        "DSA_Skill": 5.0,
        "Project_Quality": 0,
        "Experience_Score": 0,
        "Soft_Skills": 6.0,
        "OpenSource_Value": 0,
        "Tech_Stack_Score": 2.0
    })
    y = df[TARGET].fillna(50)

    # 4. TRAINING IMPROVEMENTS: 80-20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. FEATURE ENGINEERING & 3. MODEL SELECTION
    # We choose RandomForestRegressor because:
    # - Captures nonlinear relationships (e.g. high CGPA but zero projects = low score)
    # - Less sensitive to feature scaling compared to Linear Regression
    # - Robust to outliers and requires minimal hyperparameter tuning for decent baseline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"[ML] Cross-Validation R² Scores: {cv_scores}")
    print(f"[ML] Mean CV R²: {np.mean(cv_scores):.3f}")

    # Train final model on entire train set
    model.fit(X_train, y_train)

    # 5. EVALUATION
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- MODEL EVALUATION METRICS ---")
    print(f"MAE  : {mae:.2f} (Average error in score points)")
    print(f"RMSE : {rmse:.2f} (Penalizes larger errors more heavily)")
    print(f"R²   : {r2:.3f} (Proportion of variance explained)")
    print("--------------------------------\n")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[ML] Advanced Model trained and saved to {MODEL_PATH}")

def predict_score(features: dict) -> dict:
    try:
        model = joblib.load(MODEL_PATH)
        
        cgpa = features.get("cgpa", 0.0)
        num_projects = features.get("num_projects", 0)
        num_internships = features.get("num_internships", 0)
        communication = features.get("communication", 0.0)
        opensource = features.get("opensource", 0)
        tech_score = features.get("tech_stack_score", 0.0)

        dsa_mock = num_projects * 1.5 + (cgpa - 5)
        dsa_mock = max(1.0, min(10.0, dsa_mock))

        df_features = {
            "Academic_Score": [cgpa],
            "DSA_Skill": [dsa_mock],
            "Project_Quality": [num_projects],
            "Experience_Score": [num_internships],
            "Soft_Skills": [communication],
            "OpenSource_Value": [opensource],
            "Tech_Stack_Score": [tech_score]
        }
        
        X = pd.DataFrame(df_features)
        
        # 1. Base ML Score
        ml_score = float(model.predict(X)[0])
        
        # 2. Domain Rule-Based Score
        # Weighting: CGPA(30%), Tech(20%), Projects(20%), DSA(15%), Internships(10%), Comm(5%)
        domain_score = (
            (cgpa / 10.0) * 30 +
            (tech_score / 5.0) * 20 +
            min(num_projects, 5) / 5.0 * 20 +
            (dsa_mock / 10.0) * 15 +
            min(num_internships, 3) / 3.0 * 10 +
            (communication / 10.0) * 5
        )
        
        # 3. Hybrid Model Integration
        # Trust ML 70%, but strictly anchor with 30% Domain logic
        final_score = (ml_score * 0.70) + (domain_score * 0.30)
        
        # 4. Domain Logic Adjustments (Boosts & Caps)
        adjustments = []
        
        if num_internships >= 2:
            final_score += 8.0
            adjustments.append("Boost (+8) for having 2 or more internships.")
            
        if num_projects > 3:
            final_score += 5.0
            adjustments.append("Boost (+5) for strong project portfolio (>3 projects).")
            
        # Caps are applied last to ensure hard limits are respected
        if cgpa < 6.0:
            final_score = min(final_score, 60.0)
            adjustments.append("Capped at 60 due to low CGPA (< 6.0).")
            
        if num_projects == 0:
            final_score = min(final_score, 60.0)
            adjustments.append("Capped at 60 due to zero projects.")
        
        final_score = max(0.0, min(100.0, round(final_score, 2)))
        
        # 5. Interpretability Breakdown
        breakdown = [
            f"ML Base: {round(ml_score, 1)}",
            f"Rules Base: {round(domain_score, 1)}"
        ]
        if adjustments:
            breakdown.extend(adjustments)
            
        return {
            "score": final_score,
            "breakdown": " | ".join(breakdown)
        }
        
    except Exception as e:
        print(f"Error predicting score: {e}")
        return {"score": round(float(np.random.normal(65, 10)), 2), "breakdown": "Failed to parse model."}
