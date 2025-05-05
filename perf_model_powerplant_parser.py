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
    
    # --- Define regex patterns (tried in order) ---
    patterns = []
    
    # Pattern 9:
    # Handles cases like "2x (78kN) PowerJet SaM 146"
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*\(\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s*\)\s+'
        r'(?P<manufacturer>PowerJet\s+SaM)\s+(?P<engine_code>\S+)',
        re.IGNORECASE
    ))
    
    # Pattern 7:
    # Handles cases like "4 x 185 kN P&W F117-PW-100 (PW2040) turbofans."
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
        r'(?P<manufacturer>P\s*&\s*W.*?)\s+.*?\(\s*(?P<engine_code>PW\d+)\s*\)',
        re.IGNORECASE
    ))
    
    # Pattern A (new):
    # Handles cases where no manufacturer is specified; e.g. "2 x 262kN CF6-80C2"
    patterns.append(re.compile(
        r'^(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
        r'(?P<engine_code>[A-Z0-9\-\s/]+)$',
        re.IGNORECASE
    ))
    
    # Pattern 1:
    # Standard inline thrust value with manufacturer.
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
        r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+'
        r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
        re.IGNORECASE
    ))
    
    # Pattern 8:
    # For entries that do not contain an explicit multiplier.
    # Example: "61.3 kN rated GE CF34 - 8C1 turbofans." → default number = 2.
    patterns.append(re.compile(
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP).*?'
        r'(?:rated\s+)?'
        r'(?P<manufacturer>GE|General\s+Electric)\s+'
        r'(?P<engine_code>CF34\s*-\s*8C1)',
        re.IGNORECASE
    ))
    
    # Pattern 2:
    # When the text includes a multiplier as a word.
    # Example: "two Pratt & Whitney JT8D-5"
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]?\s*'
        r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+'
        r'(?P<engine_code>[A-Z0-9\-\s/]+)',
        re.IGNORECASE
    ))
    
    # Pattern 3:
    # Another standard pattern with thrust.
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)\s*(?P<unit>kN|kW|hp|SHP)\s+'
        r'(?P<manufacturer>[A-Z][A-Za-z&\-\s]+?)\s+'
        r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
        re.IGNORECASE
    ))
    
    # Pattern 4:
    # For entries like "2x GF34-8E" (mistyped) or similar.
    patterns.append(re.compile(
        r'(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*[x×]\s*'
        r'(?P<thrust>\d+(?:[.,]\d+)?\.?)?\s*(?P<unit>kN|kW|hp|SHP)?\s*'
        r'(?P<engine_code>[A-Z0-9\-\s/]+?)(?=\s+(?:turbofan|turboprop|turbojet|engine|each)|\s*$)',
        re.IGNORECASE
    ))
    
    # Pattern 5:
    # For entries with manufacturer and code only.
    patterns.append(re.compile(
        r'(?P<manufacturer>General\s+Electric|GE)\s+'
        r'(?P<engine_code>[A-Z0-9\-\s/]+)',
        re.IGNORECASE
    ))
    
    def extract_engine_info(spec):
        """
        Try each regex pattern (in order) on the given spec text.
        Return a dictionary of extracted fields if a pattern matches; else, return None.
        """
        spec = spec.strip()
        for pat in patterns:
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
                
                # Remove trailing dot from thrust, if any.
                if thrust_val.endswith('.'):
                    thrust_val = thrust_val[:-1]
                engine_code = " ".join(engine_code.split())
                
                # Special case: correct mistyped engine code "GF34-8E" → "CF34-8E"
                if re.fullmatch(r'GF34-8E', engine_code, re.IGNORECASE):
                    engine_code = "CF34-8E"
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
                    'engine_code': engine_code
                }
        logging.warning("No match for spec: '%s'", spec)
        return None
    
    def process_powerplant_text(text):
        """
        Preprocess the 'powerplant' text and then try to extract engine info.
        Special handling:
          - If the text is "no data" (case-insensitive), return None.
          - If the text contains "Garrett AiResearch TPE 331", return None.
          - If the text starts with variant markers like "GR1:", split on "International:"
            and remove any "(with afterburner ...)" substrings, then remove the prefix.
          - If no explicit multiplier is found but the text mentions a plural form (e.g. "turbofans"),
            default the number to 2.
        """
        text = text.strip()
        if text.lower() == "no data":
            return None
        if "Garrett AiResearch TPE 331" in text:
            return None
        # Special handling for GR1: variants.
        if text.startswith("GR1:"):
            # Keep only the part before "International:" (if present).
            text = text.split("International:")[0].strip()
            # Remove any substring like "(with afterburner ...)"
            text = re.sub(r'\(with afterburner.*?\)', '', text, flags=re.IGNORECASE).strip()
            # Remove the "GR1:" prefix.
            text = re.sub(r'^GR1:\s*', '', text, flags=re.IGNORECASE)
        
        # If no explicit multiplier is present but text contains "turbofans", prepend "2 x ".
        if not re.search(r'\d+\s*[x×]', text) and re.search(r'\bturbofans?\b', text, re.IGNORECASE):
            text = "2 x " + text
        
        # Split on " or " and take the first configuration that produces a match.
        specs = re.split(r'\s+or\s+', text, flags=re.IGNORECASE)
        for spec in specs:
            info = extract_engine_info(spec)
            if info is not None:
                return info
        return None

    # Process each row of the input DataFrame.
    num_engines_list = []
    thrust_list = []
    engine_code_list = []
    manufacturer_list = []
    
    for idx, row in df.iterrows():
        text = row['powerplant']
        info = process_powerplant_text(text)
        if info:
            num_engines_list.append(info.get('number_of_engines'))
            thrust_list.append(info.get('thrust'))
            engine_code_list.append(info.get('engine_code'))
            manufacturer_list.append(info.get('manufacturer'))
        else:
            num_engines_list.append(None)
            thrust_list.append(None)
            engine_code_list.append(None)
            manufacturer_list.append(None)
    
    df_result = df.copy()
    df_result['number_of_engines'] = num_engines_list
    df_result['thrust'] = thrust_list
    df_result['engine_code'] = engine_code_list
    df_result['manufacturer'] = manufacturer_list
    return df_result