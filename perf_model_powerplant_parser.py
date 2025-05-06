import pandas as pd
import re
import logging

def perf_model_powerplant_parser(df):
    """
    Given a DataFrame with engine data in the 'powerplant' column,
    return a modified DataFrame with added columns:
      - number_of_engines
      - thrust      (extracted numeric thrust value and its unit)
      - engine_code (the model/type code)
      - manufacturer
    This function applies several regex patterns (with special-case logic)
    to accommodate a wide range of text formats.
    """
    # Configure logging.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Map textual numbers to digits.
    text_to_digit = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # --- Define regex patterns as tuples (label, compiled_pattern) ---
    patterns = [
        # Pattern 9: Handles cases like "2x (78kN) PowerJet SaM 146"
        ("Pattern 9", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*\(\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s*\)\s+'
            r'(?P<manufacturer>PowerJet\s+SaM)\s*(?P<engine_code>\S+)',
            re.IGNORECASE
        )),
        # Pattern 7: Handles cases like "4 x 185 kN P&W F117-PW-100 (PW2040) turbofans."
        ("Pattern 7", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
            r'(?P<manufacturer>P\s*&\s*W.*?)\s+.*?\(\s*(?P<engine_code>PW\d+)\s*\)',
            re.IGNORECASE
        )),
        # Pattern 8: For entries with no explicit multiplier.
        # Example: "61.3 kN rated GE CF34 - 8C1 turbofans." → default number = 2.
        ("Pattern 8", re.compile(
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP).*?'
            r'(?:rated\s+)?'
            r'(?P<manufacturer>GE|General\s+Electric)\s+'
            r'(?P<engine_code>CF34\s*-\s*8C1)',
            re.IGNORECASE
        )),
        # Pattern 1: Standard inline thrust value.
        ("Pattern 1", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+'
            r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
            re.IGNORECASE
        )),
        # Pattern 10: For cases where after the thrust unit the engine code follows with no explicit manufacturer.
        ("Pattern 10", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s*'
            r'(?P<engine_code>[A-Z][A-Za-z0-9\-\s/]+)',
            re.IGNORECASE
        )),
        # Pattern 3: Standard pattern with thrust.
        ("Pattern 3", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+'
            r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
            re.IGNORECASE
        )),
        
        # Pattern 4: For entries like "2x GF34-8E" (mistyped) or similar.
        ("Pattern 4", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<thrust>\d+(?:[.,]\d+)?\.?)?\s*(?P<unit>kN|kW|hp|SHP)?\s*'
            r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
            re.IGNORECASE
        )),
        # Pattern 5: For entries with manufacturer and code only.
        ("Pattern 5", re.compile(
            r'(?P<manufacturer>General\s+Electric|GE)\s+'
            r'(?P<engine_code>[A-Z0-9\-\s/]+)',
            re.IGNORECASE
        )),
        # Pattern 2 (modified): Textual multiplier pattern.
        # Added lookahead (?=\D) to ensure the manufacturer token does not start with a digit.
        # Example: "two Pratt & Whitney JT8D-5"
        ("Pattern 2", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]?\s*(?=\D)'
            r'(?P<manufacturer>(?!\d)[A-Z][A-Za-z&\-\s]+?)\s+'
            r'(?P<engine_code>[A-Z0-9\-\s/]+)',
            re.IGNORECASE
        )),
        # Pattern 11: Handles '2x RR Trent XWB delivering 374kN each'
        ("Pattern 11", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+(?P<engine_code>[A-Z0-9\-\s/]+?)\s+delivering\s+(?P<thrust>\d+(?:[.,]\d+)?\.?)(?P<unit>kN|kW|hp|SHP)\s*each',
            re.IGNORECASE
        )),
        # Pattern 12: Handles '2 x RR Trent XWB (430kN each)'
        ("Pattern 12", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+(?P<engine_code>[A-Z0-9\-\s/]+?)\s*\(\s*(?P<thrust>\d+(?:[.,]\d+)?\.?)(?P<unit>kN|kW|hp|SHP)\s*each\s*\)',
            re.IGNORECASE
        )),
        # Pattern 13: Handles '2 x CFM International LEAP-1B (130 kN) turbofans'
        ("Pattern 13", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+(?P<engine_code>[A-Z0-9\-\s/]+?)\s*\(\s*(?P<thrust>\d+(?:[.,]\d+)?\.?)(?P<unit>kN|kW|hp|SHP)\s*\)\s*(?:turbofan|turboprop|turbojet|engine|each)?',
            re.IGNORECASE
        )),
        # Pattern 14: Handles '2x General Electric GE90-110B1 110.100lbf(490kN) / GE90-115B1 115.540lbf (514kN)'
        ("Pattern 14", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+(?P<engine_code>[A-Z0-9\-\s/]+?)\s+(?P<lbf>\d{1,3}(?:[.,]\d{3})*(?:lbf|lbF))?\s*\(\s*(?P<thrust>\d+(?:[.,]\d+)?)(?P<unit>kN|kW|hp|SHP)\s*\)',
            re.IGNORECASE
        )),
        # Pattern 15: Handles '2 x General Electric GE90-115B 115,000 lbF (510 kN).'
        ("Pattern 15", re.compile(
            r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
            r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+(?P<engine_code>[A-Z0-9\-\s/]+?)\s+(?P<lbf>\d{1,3}(?:[.,]\d{3})*\s*lbF)\s*\(\s*(?P<thrust>\d+(?:[.,]\d+)?)(?P<unit>kN|kW|hp|SHP)\s*\)',
            re.IGNORECASE
        )),
    ]
    
    def extract_engine_info(spec):
        """
        Try each regex pattern (in order) on the spec text.
        Returns a dictionary of extracted fields—including 'regex_path' set to the pattern label—
        if a pattern matches; otherwise, returns None.
        """
        spec = spec.strip()
        for label, pat in patterns:
            m = pat.search(spec)
            if m:
                groups = m.groupdict()
                regex_used = label  # use the explicit label
                num_str = groups.get('number', None)
                if num_str:
                    num_str = num_str.lower()
                    number = int(num_str) if num_str.isdigit() else text_to_digit.get(num_str, None)
                else:
                    number = 2  # default if not provided
                
                thrust_val = groups.get('thrust', '') or ''
                unit = groups.get('unit', '') or ''
                manufacturer = groups.get('manufacturer', None)
                engine_code = groups.get('engine_code', '').strip()
                
                # Remove trailing dot in thrust.
                if thrust_val.endswith('.'):
                    thrust_val = thrust_val[:-1]
                engine_code = " ".join(engine_code.split())
                
                # Special case: correct "GF34-8E" to "CF34-8E".
                if re.fullmatch(r'GF34-8E', engine_code, re.IGNORECASE):
                    engine_code = "CF34-8E"
                    manufacturer = "General Electric"
                
                # If manufacturer is still missing (as might occur in Pattern 10), derive it from engine_code.
                if not manufacturer:
                    tokens = engine_code.split()
                    if tokens:
                        first_token = tokens[0]
                        if first_token.lower() == "garrett":
                            manufacturer = "Garrett"
                            engine_code = " ".join(tokens[1:])
                        elif first_token.upper().startswith("CF6") or first_token.upper().startswith("CFM"):
                            manufacturer = "General Electric"
                # Normalize manufacturer names.
                if manufacturer:
                    if re.search(r'P\s*&\s*W', manufacturer, re.IGNORECASE):
                        manufacturer = "Pratt & Whitney"
                    elif re.search(r'\bR[-–]?R\b', manufacturer, re.IGNORECASE):
                        manufacturer = "Rolls Royce"
                    elif re.fullmatch(r'GE', manufacturer, re.IGNORECASE):
                        manufacturer = "General Electric"
                    elif re.search(r'PowerJet\s+SaM', manufacturer, re.IGNORECASE):
                        manufacturer = "PowerJet S.A."
                return {
                    'number_of_engines': number,
                    'thrust': thrust_val + ((' ' + unit) if thrust_val and unit else ''),
                    'manufacturer': manufacturer,
                    'engine_code': engine_code,
                    'regex_path': regex_used
                }
        logging.warning("No match for spec: '%s'", spec)
        return None
    
    def extract_engine_info_all(spec):
        """
        Try all regex patterns on the spec text.
        Return a list of (score, info_dict, label) for each match.
        """
        spec = spec.strip()
        results = []
        for label, pat in patterns:
            m = pat.search(spec)
            if m:
                groups = m.groupdict()
                num_str = groups.get('number', None)
                if num_str:
                    num_str = num_str.lower()
                    number = int(num_str) if num_str.isdigit() else text_to_digit.get(num_str, None)
                else:
                    number = 2  # default if not provided
                thrust_val = groups.get('thrust', '') or ''
                unit = groups.get('unit', '') or ''
                manufacturer = groups.get('manufacturer', None)
                engine_code = groups.get('engine_code', '').strip()
                if thrust_val.endswith('.'):
                    thrust_val = thrust_val[:-1]
                engine_code = " ".join(engine_code.split())
                # Special case: correct "GF34-8E" to "CF34-8E".
                if re.fullmatch(r'GF34-8E', engine_code, re.IGNORECASE):
                    engine_code = "CF34-8E"
                    manufacturer = "General Electric"
                # If manufacturer is still missing (as might occur in Pattern 10), derive it from engine_code.
                if not manufacturer:
                    tokens = engine_code.split()
                    if tokens:
                        first_token = tokens[0]
                        if first_token.lower() == "garrett":
                            manufacturer = "Garrett"
                            engine_code = " ".join(tokens[1:])
                        elif first_token.upper().startswith("CF6") or first_token.upper().startswith("CFM"):
                            manufacturer = "General Electric"
                # Normalize manufacturer names.
                if manufacturer:
                    if re.search(r'P\s*&\s*W', manufacturer, re.IGNORECASE):
                        manufacturer = "Pratt & Whitney"
                    elif re.search(r'\bR[-–]?R\b', manufacturer, re.IGNORECASE):
                        manufacturer = "Rolls Royce"
                    elif re.fullmatch(r'GE', manufacturer, re.IGNORECASE):
                        manufacturer = "General Electric"
                    elif re.search(r'PowerJet\s+SaM', manufacturer, re.IGNORECASE):
                        manufacturer = "PowerJet S.A."
                info = {
                    'number_of_engines': number,
                    'thrust': thrust_val + ((' ' + unit) if thrust_val and unit else ''),
                    'manufacturer': manufacturer,
                    'engine_code': engine_code,
                    'regex_path': label
                }
                # Score: count non-empty fields (except regex_path)
                score = sum(1 for k in ['number_of_engines','thrust','manufacturer','engine_code'] if info[k])
                results.append((score, info, label))
        return results

    def process_powerplant_text_best(text):
        text = text.strip()
        if text.lower() == "no data":
            return None
        if "Garrett AiResearch TPE 331" in text:
            return None
        if text.startswith("GR1:"):
            text = text.split("International:")[0].strip()
            text = re.sub(r'\(with afterburner.*?\)', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^GR1:\s*', '', text, flags=re.IGNORECASE)
        if not re.search(r'\d+\s*[x×]', text) and re.search(r'\bturbofans?\b', text, re.IGNORECASE):
            text = "2 x " + text
        specs = re.split(r'\s+or\s+', text, flags=re.IGNORECASE)
        best_score = -1
        best_info = None
        for spec in specs:
            results = extract_engine_info_all(spec)
            for score, info, label in results:
                if score > best_score:
                    best_score = score
                    best_info = info
        return best_info

    # Process each row in the DataFrame.
    num_engines_list = []
    thrust_list = []
    engine_code_list = []
    manufacturer_list = []
    regex_path_list = []
    for idx, row in df.iterrows():
        text = row['powerplant']
        info = process_powerplant_text_best(text)
        if info:
            num_engines_list.append(info.get('number_of_engines'))
            thrust_list.append(info.get('thrust'))
            engine_code_list.append(info.get('engine_code'))
            manufacturer_list.append(info.get('manufacturer'))
            regex_path_list.append(info.get('regex_path'))
        else:
            num_engines_list.append(None)
            thrust_list.append(None)
            engine_code_list.append(None)
            manufacturer_list.append(None)
            regex_path_list.append(None)
    df_result = df.copy()
    df_result['number_of_engines'] = num_engines_list
    df_result['thrust'] = thrust_list
    df_result['engine_code'] = engine_code_list
    df_result['manufacturer'] = manufacturer_list
    df_result['regex_path'] = regex_path_list
    return df_result
