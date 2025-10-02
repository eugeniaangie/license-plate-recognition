import re

# Complete Indonesian license plate area codes (all provinces)
VALID_AREA_CODES = [
    # Single letter
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    
    # Double letter - Sumatra
    'AA', 'AB', 'AD', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP', 'BR', 'BT',
    
    # Double letter - Java
    'DA', 'DB', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DT', 
    'EA', 'EB', 'ED', 'EG', 'EH',
    'FA', 'FB', 'FD', 'FE', 'FG', 'FH',
    'GA', 'GB', 'GC', 'GD', 'GE', 'GH', 'GK', 'GL', 'GM', 'GN', 'GP', 'GS', 'GT', 'GU', 'GW',
    
    # Double letter - Kalimantan
    'DA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG', 'KH', 'KJ', 'KK', 'KL', 'KM', 'KN', 'KO', 'KP', 'KQ', 'KR', 'KS', 'KT', 'KU',
    
    # Double letter - Sulawesi
    'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW',
    
    # Double letter - Others
    'PA', 'PB', 'PD', 'PE', 'PG', 'PH', 'PK', 'PL', 'PM', 'PN', 'PP',
]

# Remove duplicates and sort
VALID_AREA_CODES = sorted(list(set(VALID_AREA_CODES)))

def fix_letters(s):
    """
    Fix angka yang kebaca di area huruf
    Used for area code and suffix letters
    """
    replacements = {
        '0': 'O',
        '1': 'I',
        '2': 'Z',
        '3': 'E',
        '4': 'A',
        '5': 'S',
        '6': 'G',
        '7': 'T',
        '8': 'B',
        '9': 'P'
    }
    result = s
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result

def fix_numbers(s):
    """
    Fix huruf yang kebaca di area angka
    Used for number section
    """
    replacements = {
        'O': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'E': '3',
        'A': '4',
        'S': '5',
        'G': '6',
        'T': '7',
        'B': '8',
        'H': '8',
        'P': '9',
        'D': '0',
        'Q': '0'
    }
    result = s
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result

def validate_format(text):
    """
    Validate Indonesian plate format
    Format: [Area 1-2 letters][Numbers 1-4 digits][Suffix 0-3 letters]
    Examples: B1234ABC, DD8765XY, L123, F4567
    """
    # Pattern untuk Indonesian plates
    patterns = [
        r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$',  # Standard format
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            area = match.group(1)
            number = match.group(2)
            suffix = match.group(3) if len(match.groups()) >= 3 else ""
            
            # Validate area code
            if area in VALID_AREA_CODES:
                # Validate number length (typically 3-4 digits, but can be 1-4)
                if len(number) >= 1 and len(number) <= 4:
                    # Validate suffix length (0-3 letters)
                    if len(suffix) <= 3:
                        return True, area, number, suffix
    
    return False, None, None, None

def post_process_plate(text):
    """
    Clean & fix OCR result sesuai format plat Indonesia
    Returns: cleaned text or None if invalid
    """
    if not text:
        return None
    
    # 1. Basic cleaning
    text = text.upper().strip()
    text = text.replace(" ", "").replace("-", "").replace(".", "")
    text = ''.join(c for c in text if c.isalnum())
    
    # Skip if too short or too long
    if len(text) < 3 or len(text) > 10:
        return None
    
    # 2. Check if starts with number (missing area code!)
    if text[0].isdigit():
        # Plate HARUS start dengan letter, berarti area code missing
        # Return as-is tapi mark untuk manual check
        # Post-processing nanti bisa add common area codes
        return text  # Will be marked invalid by validate_format
    
    # 3. Try multiple parsing strategies
    strategies = [
        # Strategy 1: Assume first 1-2 chars are area code
        lambda t: parse_strategy_1(t),
        # Strategy 2: Find pattern AA1234BB
        lambda t: parse_strategy_2(t),
        # Strategy 3: Mixed characters, need heavy fixing
        lambda t: parse_strategy_3(t),
    ]
    
    for strategy in strategies:
        result = strategy(text)
        if result:
            return result
    
    # Fallback: return cleaned text (might still be useful)
    return text if len(text) >= 3 else None

def parse_strategy_1(text):
    """
    Strategy 1: Parse assuming clear segments
    Try 1-letter then 2-letter area codes
    """
    # Try 2-letter area code first
    if len(text) >= 4:
        area_candidate = text[:2]
        rest = text[2:]
        
        # Split rest into numbers and suffix
        match = re.match(r'^(\d+)([A-Z]*)$', rest)
        if match:
            number = match.group(1)
            suffix = match.group(2)
            
            if area_candidate in VALID_AREA_CODES:
                return f"{area_candidate}{number}{suffix}"
    
    # Try 1-letter area code
    if len(text) >= 2:
        area_candidate = text[0]
        rest = text[1:]
        
        match = re.match(r'^(\d+)([A-Z]*)$', rest)
        if match:
            number = match.group(1)
            suffix = match.group(2)
            
            if area_candidate in VALID_AREA_CODES:
                return f"{area_candidate}{number}{suffix}"
    
    return None

def parse_strategy_2(text):
    """
    Strategy 2: Use regex patterns with character fixing
    """
    # Try to find pattern: letters, numbers, letters
    patterns = [
        r'^([A-Z0-9]{1,2})([0-9A-Z]{1,4})([A-Z0-9]{0,3})$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            area_raw = match.group(1)
            number_raw = match.group(2)
            suffix_raw = match.group(3) if len(match.groups()) >= 3 else ""
            
            # Fix characters
            area = fix_letters(area_raw)
            number = fix_numbers(number_raw)
            suffix = fix_letters(suffix_raw)
            
            # Validate
            result = f"{area}{number}{suffix}"
            is_valid, _, _, _ = validate_format(result)
            
            if is_valid:
                return result
    
    return None

def parse_strategy_3(text):
    """
    Strategy 3: Aggressive character fixing for mixed strings
    """
    # Assume format: 1-2 letters, then numbers, then optional letters
    
    # Try to split by detecting where numbers start
    first_digit_idx = -1
    for i, c in enumerate(text):
        if c.isdigit():
            first_digit_idx = i
            break
    
    if first_digit_idx == -1:
        return None
    
    # Extract potential area code
    area_raw = text[:first_digit_idx]
    if len(area_raw) < 1 or len(area_raw) > 2:
        return None
    
    rest = text[first_digit_idx:]
    
    # Find where numbers end
    last_digit_idx = -1
    for i in range(len(rest)-1, -1, -1):
        if rest[i].isdigit() or rest[i] in 'OILZEASGTBHPD':  # Characters that can be numbers
            last_digit_idx = i
            break
    
    if last_digit_idx == -1:
        number_raw = rest
        suffix_raw = ""
    else:
        number_raw = rest[:last_digit_idx+1]
        suffix_raw = rest[last_digit_idx+1:]
    
    # Fix characters
    area = fix_letters(area_raw)
    number = fix_numbers(number_raw)
    suffix = fix_letters(suffix_raw)
    
    # Validate
    result = f"{area}{number}{suffix}"
    is_valid, _, _, _ = validate_format(result)
    
    if is_valid:
        return result
    
    return None

def confidence_score(text):
    """
    Calculate confidence score for a plate text
    Based on format validation and characteristics
    """
    if not text:
        return 0.0
    
    score = 0.0
    
    # Check format validity
    is_valid, area, number, suffix = validate_format(text)
    
    if is_valid:
        score += 50.0
        
        # Bonus for common number lengths
        if len(number) == 4:
            score += 20.0
        elif len(number) == 3:
            score += 15.0
        
        # Bonus for having suffix
        if len(suffix) > 0:
            score += 10.0
        
        # Bonus for common area codes
        common_areas = ['B', 'D', 'F', 'E', 'AA', 'AB', 'L', 'N', 'DD', 'DK']
        if area in common_areas:
            score += 10.0
        
        # Length check
        if 5 <= len(text) <= 9:
            score += 10.0
    
    return min(score, 100.0)