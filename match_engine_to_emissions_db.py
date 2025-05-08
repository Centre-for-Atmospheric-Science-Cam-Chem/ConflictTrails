import numpy as np
import pandas as pd
from difflib import SequenceMatcher, get_close_matches

def match_engine_to_emissions_db(performance_models, engine_models, code_weight=0.6, thrust_weight=0.4, thrust_kn_tolerance=15.0):
    # this function skips engine models that have 'SHP' or 'HP' in the thrust column, corresponding to turboshaft (geared)
    # or turboprop (ungeared engines)
    def parse_thrust_kn(thrust_str):
        if not isinstance(thrust_str, str):
            return None
        thrust_str = thrust_str.replace('KN', 'kN').replace('Kn', 'kN').replace('kn', 'kN')
        if 'kN' in thrust_str:
            try:
                return float(thrust_str.replace('kN', '').replace(',', '.').strip())
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
    for idx, row in performance_models.iterrows():
        thrust = row.get('thrust', '')
        engine_code = str(row.get('engine_code', '')).strip()
        manufacturer = str(row.get('manufacturer', '')).strip()
        if isinstance(thrust, str) and ('SHP' in thrust or 'HP' in thrust):
            matched_ids.append(None)
            continue
        thrust_kn = parse_thrust_kn(thrust)
        # If thrust is missing, but there is an exact match between engine_code and Engine Identification, allow a match
        if (not thrust_kn or pd.isna(thrust_kn)) and engine_code:
            exact_matches = engine_models[engine_models['Engine Identification'].str.strip().str.lower() == engine_code.lower()]
            if not exact_matches.empty:
                matched_ids.append(exact_matches.iloc[0]['Engine Identification'])
                continue
        # If thrust is missing, but there is a match within one character between engine_code and Engine Identification, allow a match
        if (not thrust_kn or pd.isna(thrust_kn)) and engine_code:
            def is_within_one_char(a, b):
                if a == b:
                    return True
                if abs(len(a) - len(b)) > 1:
                    return False
                # Check Levenshtein distance 1 (insert, delete, or substitute one char)
                from difflib import SequenceMatcher
                sm = SequenceMatcher(None, a, b)
                return sm.quick_ratio() > 0.9 and sum(x[0] != x[1] for x in zip(a, b)) <= 1
            engine_code_lower = engine_code.lower()
            candidates = engine_models['Engine Identification'].str.strip().str.lower().tolist()
            for idx2, candidate in enumerate(candidates):
                if is_within_one_char(engine_code_lower, candidate):
                    matched_ids.append(engine_models.iloc[idx2]['Engine Identification'])
                    break
            else:
                # If no match within one char, continue to normal scoring
                pass
            if len(matched_ids) > idx:
                continue
        candidates = []
        for _, em_row in engine_models.iterrows():
            em_code = em_row['Engine Identification']
            em_thrust = em_row['Rated Thrust (kN)']
            code_sim = string_similarity(engine_code.lower(), em_code.lower()) if engine_code else 0
            # Thrust similarity (normalized: 1 for perfect match, 0 for >50kN diff)
            if thrust_kn is not None and pd.notnull(em_thrust):
                thrust_diff = abs(thrust_kn - em_thrust)
                thrust_sim = max(0, 1 - (thrust_diff / 50.0))
            else:
                thrust_diff = None
                thrust_sim = 0
            score = code_weight * code_sim + thrust_weight * thrust_sim
            candidates.append((score, thrust_diff, em_code))
        candidates.sort(reverse=True, key=lambda x: x[0])
        filtered = [c for c in candidates if c[1] is not None and c[1] <= thrust_kn_tolerance]
        selected_id = None
        if filtered:
            engine_code_lower = engine_code.lower().rstrip(' series').strip()
            for _, _, em_code in filtered:
                em_code_lower = em_code.lower().rstrip(' series').strip()
                if em_code_lower == engine_code_lower or em_code_lower.startswith(engine_code_lower):
                    selected_id = em_code
                    break
            if not selected_id:
                selected_id = filtered[0][2]
        matched_ids.append(selected_id)
    result = performance_models.copy()
    result['matched_engine_id'] = matched_ids
    # Add matched engine thrust column
    matched_thrusts = []
    matched_uids = []
    for mid in matched_ids:
        if mid is not None and mid in engine_models['Engine Identification'].values:
            thrust_val = engine_models.loc[engine_models['Engine Identification'] == mid, 'Rated Thrust (kN)'].values
            uid_val = engine_models.loc[engine_models['Engine Identification'] == mid, 'UID No'].values
            matched_thrusts.append(thrust_val[0] if len(thrust_val) > 0 else None)
            matched_uids.append(uid_val[0] if len(uid_val) > 0 else None)
        else:
            matched_thrusts.append(None)
            matched_uids.append(None)
    result['matched_engine_thrust_kn'] = matched_thrusts
    result['matched_engine_uid_no'] = matched_uids
    return result