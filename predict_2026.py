import pandas as pd
import joblib


print("🔄 Loading Model ...")
model_bundle = joblib.load('f1_model_v3_multiyear.pkl')
model = model_bundle['model']
model_cols = model_bundle['model_columns']
print("✅ Model Loaded Successfully!")

team_proxy_map = {
    # The Champions
    'Red Bull': 'Red Bull Racing',
    'RBR': 'Red Bull Racing',
    
    # The Sister Team
    'VCARB': 'RB',
    'Racing Bulls': 'RB',
    'AlphaTauri': 'RB',
    
    # Rebrands & New Teams
    'Audi': 'Kick Sauber',       # Audi = Sauber
    'Cadillac': 'Haas F1 Team'   # Cadillac = Haas (Proxy)
}

driver_proxy_map = {
    'LIN': 'LAW'  # Arvid Lindblad -> Liam Lawson (The Rookie Benchmark)
}

def predict_race(driver, team, grid, circuit_type):

    eff_team = team_proxy_map.get(team, team)
    eff_driver = driver_proxy_map.get(driver, driver)
    
    input_data = pd.DataFrame(0, index=[0], columns=model_cols)
    input_data['GridPosition'] = grid
    
    if f"Team_{eff_team}" in model_cols:
        input_data[f"Team_{eff_team}"] = 1
    else:
        print(f"⚠️  Unknown Team: {eff_team}")

    if f"Driver_{eff_driver}" in model_cols:
        input_data[f"Driver_{eff_driver}"] = 1
    else:
        print(f"⚠️  Unknown Driver: {eff_driver} (Predicting based on Car only)")

    if f"Circuit_{circuit_type}" in model_cols:
        input_data[f"Circuit_{circuit_type}"] = 1
        
    prob = model.predict_proba(input_data)[0][1]
    
    note = ""
    if eff_team != team: note += f" [Team mapped to {eff_team}]"
    if eff_driver != driver: note += f" [Driver mapped to {eff_driver}]"
    
    print(f"🏎️  {driver} | {team} | P{grid} ({circuit_type})")
    print(f"📊 Podium Chance: {prob:.1%}{note}")
    print("-" * 40)


# Scenario 1: Max Verstappen
predict_race('VER', 'Red Bull', 1, 'Street')

# Scenario 2: Arvid Lindblad (The Rookie)
predict_race('LIN', 'VCARB', 12, 'Street')

# Scenario 3: Nico Hulkenberg (The Audi Era)
predict_race('HUL', 'Audi', 10, 'Street')

# Scenario 4: Lando Norris
predict_race('NOR', 'McLaren', 2, 'Street')