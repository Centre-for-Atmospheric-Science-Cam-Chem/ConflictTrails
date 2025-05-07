import numpy as np
import pandas as pd
from difflib import SequenceMatcher, get_close_matches

def match_engine_to_emissions_db(performance_models_debug, engine_models, manufacturer_kn_window=5.0, code_weight=0.7, thrust_weight=0.3):
    # this function skips engine models that have 'SHP' or 'HP' in the thrust column, corresponding to turboshaft (geared)
    # or turboprop (ungeared engines)
    def parse_thrust_kn(thrust_str):
        if not isinstance(thrust_str, str):
            return None
        if 'kN' in thrust_str:
            try:
                return float(thrust_str.replace('kN', '').replace(',', '').strip())
            except Exception:
                return None
        return None

    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    engine_models = engine_models.copy()
    engine_models['Engine Identification'] = engine_models['Engine Identification'].astype(str)
    engine_models['Manufacturer'] = engine_models['Manufacturer'].astype(str)
    engine_models['Rated Thrust (kN)'] = pd.to_numeric(engine_models['Rated Thrust (kN)'], errors='coerce')

    matched_ids = []
    for idx, row in performance_models_debug.iterrows():
        thrust = row.get('thrust', '')
        engine_code = str(row.get('engine_code', '')).strip()
        manufacturer = str(row.get('manufacturer', '')).strip()
        if isinstance(thrust, str) and ('SHP' in thrust or 'HP' in thrust):
            matched_ids.append(None)
            continue
        thrust_kn = parse_thrust_kn(thrust)
        best_score = -1
        best_id = None
        for _, em_row in engine_models.iterrows():
            em_code = em_row['Engine Identification']
            em_thrust = em_row['Rated Thrust (kN)']
            # String similarity (engine code)
            code_sim = string_similarity(engine_code.lower(), em_code.lower()) if engine_code else 0
            # Thrust similarity (normalized: 1 for perfect match, 0 for >50kN diff)
            if thrust_kn is not None and pd.notnull(em_thrust):
                thrust_diff = abs(thrust_kn - em_thrust)
                thrust_sim = max(0, 1 - (thrust_diff / 50.0))
            else:
                thrust_sim = 0
            # Weighted score
            score = code_weight * code_sim + thrust_weight * thrust_sim
            if score > best_score:
                best_score = score
                best_id = em_code
        matched_ids.append(best_id)
    result = performance_models_debug.copy()
    result['matched_engine_id'] = matched_ids
    return result