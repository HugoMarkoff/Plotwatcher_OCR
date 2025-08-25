#!/usr/bin/env python3
"""
Plotwatcher Camera Trap OCR - Enhanced with Timelapse Sequence Detection
Handles Plotwatcher-specific patterns, filename parsing, and timelapse logic
"""

import os
import re
import cv2
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

# OCR engine imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class PlotwatcherOCR:
    """Specialized OCR processor for Plotwatcher cameras"""
    
    def __init__(self, engine: str = "easyocr", confidence: float = 0.3):
        self.engine = engine.lower()
        self.confidence = confidence
        self.reader = None
        
        # Initialize OCR engine
        if self.engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            self.reader = PaddleOCR(use_textline_orientation=False, lang='en')
            logger.info("✅ PaddleOCR initialized for Plotwatcher")
        elif EASYOCR_AVAILABLE:
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            self.engine = "easyocr"
            logger.info("✅ EasyOCR initialized for Plotwatcher")
        else:
            raise RuntimeError("No OCR engine available")
    
    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract text from Plotwatcher image"""
        if not Path(image_path).exists():
            return []
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            
            # Resize if too large
            if w > 1200:
                scale = 1200 / w
                img = cv2.resize(img, (1200, int(h * scale)))
                h, w = img.shape[:2]
            
            # Plotwatcher-specific regions (they typically have info at bottom)
            regions = [
                ("bottom", img[int(h * 0.85):, :]),  # Bottom region for timestamp
                ("bottom_wide", img[int(h * 0.75):, :]),  # Wider bottom region
                ("full", img)  # Full image as fallback
            ]
            
            results = []
            for region_name, region_img in regions:
                if region_img.size == 0:
                    continue
                
                # Try normal processing first
                region_results = self._process_region(region_img, region_name)
                
                # If failed, try with preprocessing
                if not region_results:
                    processed_img = self._preprocess_plotwatcher_image(region_img)
                    region_results = self._process_region(processed_img, f"{region_name}_processed")
                
                results.extend(region_results)
            
            return self._dedupe_results(results)
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return []
    
    def _process_region(self, region_img, region_name: str) -> List[Dict[str, Any]]:
        """Process single region with OCR"""
        results = []
        
        try:
            if self.engine == "paddleocr":
                ocr_results = self.reader.predict(input=region_img)
                for result in ocr_results:
                    if hasattr(result, 'rec_texts') and hasattr(result, 'rec_scores'):
                        for text, conf in zip(result.rec_texts, result.rec_scores):
                            if text.strip() and conf > self.confidence:
                                results.append({
                                    'text': text.strip(),
                                    'confidence': float(conf),
                                    'region': region_name,
                                    'engine': 'paddleocr'
                                })
            else:  # EasyOCR
                ocr_results = self.reader.readtext(region_img, detail=True)
                for bbox, text, conf in ocr_results:
                    if text.strip() and conf > self.confidence:
                        results.append({
                            'text': text.strip(),
                            'confidence': float(conf),
                            'region': region_name,
                            'bbox': bbox,
                            'engine': 'easyocr'
                        })
        except Exception as e:
            logger.debug(f"OCR processing failed for {region_name}: {e}")
        
        return results
    
    def _preprocess_plotwatcher_image(self, img):
        """Preprocess image specifically for Plotwatcher text"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Enhance contrast for white text on dark background (common in Plotwatcher)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _dedupe_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen = set()
        unique = []
        
        for result in sorted(results, key=lambda x: x.get('confidence', 0), reverse=True):
            text_key = result['text'].lower().strip()
            if text_key not in seen and len(text_key) > 0:
                seen.add(text_key)
                unique.append(result)
        
        return unique

class PlotwatcherPatterns:
    """Plotwatcher-specific pattern matching"""
    
    def __init__(self):
        self.date_patterns = [
            # Compressed date patterns: 00773072020 -> 07/30/2020
            r'0*(\d{2})(\d{2})(\d{2})(\d{4})',                     # 00773072020
            r'[\(]?0*(\d{2})(\d{2})(\d{2})(\d{4})[\)]?',          # (08700172020)
            r'0*(\d{2})(\d{2})0(\d{2})(\d{4})\d*',                # 007730072020
            # YYMMDD format
            r'(\d{2})(\d{2})(\d{2})',                             # 200731
        ]
        
        self.time_patterns = [
            # Handle extra leading zeros: 006:36:14 -> 06:36:14
            r'[\(]?0*(\d{1,2})[\s\:\.]*(\d{2})[\s\:\.]*(\d{2})[\)]?',
            # Standard patterns
            r'(\d{1,2})[\:\.\*](\d{2})[\:\.\*](\d{2})',            # 16:20:42, 14.43.12
            r'(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})',             # Spaced colons
            # Broken time patterns like '007 :19:20'
            r'0*(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})',           # 007 :19:20 -> 07:19:20
        ]
        
        self.battery_patterns = [
            r'(\d{1,3})\*',           # 77* -> 77%
            r'(\d{1,3})\%',           # 77% -> 77%
            r'(\d{1,3})/',            # 70/ -> 70%
            r'\b(\d{2,3})\b'          # 717, 777, 747 -> 71%, 77%, 74%
        ]
        
        # Temperature patterns for Plotwatcher
        self.temperature_patterns = [
            # Tilde as minus sign (highest priority)
            r'~(\d{1,2})\s*[°]?([CcFf])',           # ~3C, ~10°F
            r'\(~(\d{1,2})\s*[°]?([CcFf])\)',       # (~3C)
            r'~(\d{1,2})\s*([CcFf])\b',             # ~5C
            # Standard patterns
            r'(\d{1,3})\s*[°]([CcFf])(?!\w)',        # 72°F
            r'\b(\d{1,3})([CcFf])\b',                # 72F
            r"(\d{1,3})'([CcFf])",                   # 72'F
            r'(\d{1,3})\*\s*([CcFf])',              # 72*F
            # OCR errors where last digit = degree symbol
            r'(\d{2})[6o0]\s*[Ff]\b',               # 776F → 77°F
            r'(\d{2})[68]\s*[Ff]\b',                # 558F → 55°F
            r'(\d{2})[6o0]\s*[Cc]\b',               # 266C → 26°C
        ]

    def extract_date_from_texts(self, texts: List[str]) -> Optional[str]:
        """
        Extract OCR date from texts using robust MM/DD/YYYY parsing.
        Accepts 10–12 character strings with separators misread.
        """
        current_year = datetime.now().year

        for text in texts:
            candidate = text.strip()
            no_spaces = re.sub(r'\s+', '', candidate)

            # Case 1: Exactly 10 characters (MM?DD?YYYY)
            if len(no_spaces) == 10:
                digits = re.sub(r'[^0-9]', '', no_spaces)
                if len(digits) == 8:
                    mm, dd, yyyy = digits[0:2], digits[2:4], digits[4:8]
                    try:
                        month, day, year = int(mm), int(dd), int(yyyy)
                        if 2000 <= year <= current_year and 1 <= month <= 12 and 1 <= day <= 31:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except ValueError:
                        continue

            # Case 2: 11 characters → try dropping first char
            elif len(no_spaces) == 11:
                trimmed = no_spaces[1:]
                digits = re.sub(r'[^0-9]', '', trimmed)
                if len(digits) == 8:
                    mm, dd, yyyy = digits[0:2], digits[2:4], digits[4:8]
                    try:
                        month, day, year = int(mm), int(dd), int(yyyy)
                        if 2000 <= year <= current_year and 1 <= month <= 12 and 1 <= day <= 31:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except ValueError:
                        continue

            # Case 3: 12 characters → try dropping first or last char
            elif len(no_spaces) == 12:
                for variant in [no_spaces[1:], no_spaces[:-1]]:
                    digits = re.sub(r'[^0-9]', '', variant)
                    if len(digits) == 8:
                        mm, dd, yyyy = digits[0:2], digits[2:4], digits[4:8]
                        try:
                            month, day, year = int(mm), int(dd), int(yyyy)
                            if 2000 <= year <= current_year and 1 <= month <= 12 and 1 <= day <= 31:
                                return f"{year:04d}-{month:02d}-{day:02d}"
                        except ValueError:
                            continue

        return None

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """🔧 FIXED: Parse Plotwatcher filename with correct pattern identification"""
        result = {
            'filename_date': None,
            'sequence_number': None,
            'sequence_identifier': None,
            'frame_id': None,
            'has_timelapse': False,
            'filename_format': 'unknown'
        }
        
        # Pattern 1: YYMMDDAA_NNNNNN.jpg - Pure numbers after underscore = sequence
        pattern1 = r'^(\d{6})([A-Z]{2})_(\d+)\.jpg$'
        match1 = re.match(pattern1, filename, re.IGNORECASE)
        if match1:
            date_part = match1.group(1)  # 200922
            aa_part = match1.group(2)    # AA
            last_part = match1.group(3)  # 001776 or Frame3798
            
            # Check if last part is pure numbers
            if last_part.isdigit():
                # Pure numbers = sequence number
                try:
                    year_part = int(date_part[:2])
                    month = int(date_part[2:4])
                    day = int(date_part[4:6])
                    
                    # Convert YY to full year
                    year = 2000 + year_part if year_part < 50 else 1900 + year_part
                    
                    if self._validate_date_components(year, month, day):
                        result.update({
                            'filename_date': f"{year:04d}-{month:02d}-{day:02d}",
                            'sequence_number': int(last_part),
                            'has_timelapse': True,
                            'filename_format': 'YYMMDDAA_NNNNNN'
                        })
                        return result
                except ValueError:
                    pass
            else:
                # Mixed characters = frame ID (not timelapse)
                try:
                    year_part = int(date_part[:2])
                    month = int(date_part[2:4])
                    day = int(date_part[4:6])
                    
                    year = 2000 + year_part if year_part < 50 else 1900 + year_part
                    
                    if self._validate_date_components(year, month, day):
                        result.update({
                            'filename_date': f"{year:04d}-{month:02d}-{day:02d}",
                            'frame_id': last_part,
                            'has_timelapse': False,
                            'filename_format': 'YYMMDDAA_FrameXXXX'
                        })
                        return result
                except ValueError:
                    pass
        
        # Pattern 2: YYMMDDAA_ID_NNNNNN.jpg - Mixed characters + numbers = ID + sequence
        pattern2 = r'^(\d{6})([A-Z]{2})_([^_]+)_(\d+)\.jpg$'
        match2 = re.match(pattern2, filename, re.IGNORECASE)
        if match2:
            date_part = match2.group(1)    # 220720
            aa_part = match2.group(2)      # AA
            identifier = match2.group(3)   # Frame4911
            sequence_part = match2.group(4) # 000175
            
            # Last part must be pure numbers for this pattern
            if sequence_part.isdigit():
                try:
                    year_part = int(date_part[:2])
                    month = int(date_part[2:4])
                    day = int(date_part[4:6])
                    
                    year = 2000 + year_part if year_part < 50 else 1900 + year_part
                    
                    if self._validate_date_components(year, month, day):
                        result.update({
                            'filename_date': f"{year:04d}-{month:02d}-{day:02d}",
                            'sequence_number': int(sequence_part),
                            'sequence_identifier': identifier,
                            'has_timelapse': True,
                            'filename_format': 'YYMMDDAA_ID_NNNNNN'
                        })
                        return result
                except ValueError:
                    pass
        
        return result
    
    def clean_plotwatcher_text(self, text: str) -> str:
        """Clean Plotwatcher-specific OCR text"""
        if not text:
            return text
        
        cleaned = text.strip()
        
        # Remove spaces around time separators but preserve structure
        cleaned = re.sub(r'(\d)\s+:\s*(\d)', r'\1:\2', cleaned)
        cleaned = re.sub(r'(\d)\s*:\s*(\d{2})\s*\.\s*(\d{2})', r'\1:\2:\3', cleaned)
        cleaned = re.sub(r'0*(\d{1,2})\s*:\s*(\d{2})\s*\.\s*(\d{2})', r'\1:\2:\3', cleaned)
        
        # Clean parentheses from time patterns
        cleaned = re.sub(r'^\(([0-9:\.]+)\)$', r'\1', cleaned)
        
        return cleaned
    
    def extract_date(self, texts: List[str]) -> Optional[str]:
        """Extract date from Plotwatcher texts"""
        for text in texts:
            cleaned_text = self.clean_plotwatcher_text(text)
            
            for pattern in self.date_patterns:
                date_match = re.search(pattern, cleaned_text)
                if date_match:
                    try:
                        groups = date_match.groups()
                        
                        if len(groups) == 4:
                            # Handle patterns like 00773072020
                            comp1 = int(groups[0])  # Could be month or day
                            comp2 = int(groups[1])  # Could be day or month
                            comp3 = int(groups[2])  # Might be month
                            year = int(groups[3])   # Year
                            
                            # Try different interpretations
                            interpretations = [
                                (comp3, comp1, year),  # month=comp3, day=comp1
                                (comp1, comp2, year),  # month=comp1, day=comp2
                                (comp2, comp1, year),  # month=comp2, day=comp1
                            ]
                            
                            for month, day, yr in interpretations:
                                if self._validate_date_components(yr, month, day):
                                    return f"{yr:04d}-{month:02d}-{day:02d}"
                                    
                        elif len(groups) == 3:
                            # Handle YYMMDD format
                            year_part = int(groups[0])
                            month = int(groups[1])
                            day = int(groups[2])
                            
                            # Convert YY to full year
                            if year_part < 50:
                                year = 2000 + year_part
                            else:
                                year = 1900 + year_part
                            
                            if self._validate_date_components(year, month, day):
                                return f"{year:04d}-{month:02d}-{day:02d}"
                                
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def extract_time(self, texts: List[str]) -> Optional[str]:
        """
        Robust OCR time extractor.
        Rules:
        - If string length == 8 → always HH:MM:SS (first 2 = HH, 4-5 = MM, 7-8 = SS)
        - If string length == 7 → assume H:MM:SS (single digit hour)
        - If string length == 9 and starts with '0' or '(' → drop first char and parse as 8
        """
        for text in texts:
            candidate = text.strip()
            no_spaces = re.sub(r'\s+', '', candidate)

            # Case 1: Exactly 8 characters → always HH:MM:SS
            if len(no_spaces) == 8:
                try:
                    hh = no_spaces[0:2]
                    mm = no_spaces[3:5]
                    ss = no_spaces[6:8]
                    hour, minute, second = int(hh), int(mm), int(ss)
                    if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                        return f"{hour:02d}:{minute:02d}:{second:02d}"
                except ValueError:
                    continue

            # Case 2: Exactly 7 characters → H:MM:SS
            elif len(no_spaces) == 7:
                try:
                    hh = no_spaces[0:1]
                    mm = no_spaces[2:4]
                    ss = no_spaces[5:7]
                    hour, minute, second = int(hh), int(mm), int(ss)
                    if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                        return f"{hour:02d}:{minute:02d}:{second:02d}"
                except ValueError:
                    continue

            # Case 3: Exactly 9 characters → drop first char if '0' or '('
            elif len(no_spaces) == 9 and (no_spaces.startswith("0") or no_spaces.startswith("(")):
                trimmed = no_spaces[1:]
                if len(trimmed) == 8:
                    try:
                        hh = trimmed[0:2]
                        mm = trimmed[3:5]
                        ss = trimmed[6:8]
                        hour, minute, second = int(hh), int(mm), int(ss)
                        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                            return f"{hour:02d}:{minute:02d}:{second:02d}"
                    except ValueError:
                        continue

        return None
        
    def _parse_time_candidate(self, candidate: str, length: int) -> Optional[str]:
        """Parse a 7 or 8 character candidate as HH:MM:SS"""
        try:
            if length == 8:
                # Format: HH?MM?SS where ? can be any separator
                hour_str = candidate[0:2]
                sep1 = candidate[2]
                minute_str = candidate[3:5]
                sep2 = candidate[5]
                second_str = candidate[6:8]
                
                # Separators should not be digits
                if sep1.isdigit() or sep2.isdigit():
                    return None
                    
            elif length == 7:
                # Format: H?MM?SS (single digit hour)
                hour_str = candidate[0:1]
                sep1 = candidate[1]
                minute_str = candidate[2:4]
                sep2 = candidate[4]
                second_str = candidate[5:7]
                
                # Separators should not be digits
                if sep1.isdigit() or sep2.isdigit():
                    return None
            else:
                return None
            
            # Validate all time components are digits
            if not (hour_str.isdigit() and minute_str.isdigit() and second_str.isdigit()):
                return None
            
            hour = int(hour_str)
            minute = int(minute_str)
            second = int(second_str)
            
            # Validate time ranges
            if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                return f"{hour:02d}:{minute:02d}:{second:02d}"
                
        except (ValueError, IndexError):
            pass
        
        return None
        
    def extract_battery_level(self, texts: List[str]) -> Optional[str]:
        """🔧 FIXED: Extract battery level with strict 2-3 digit rules"""
        for text in texts:
            # Clean text - remove all non-alphanumeric characters first
            cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper())
            
            # Skip if it contains C or F (likely temperature)
            if 'C' in cleaned or 'F' in cleaned:
                continue
            
            # Now extract only digits
            digits_only = re.sub(r'[^0-9]', '', cleaned)
            
            # Apply strict 2-3 digit rule
            if len(digits_only) == 2:
                battery_value = int(digits_only)
                if 10 <= battery_value <= 99:
                    return f"{battery_value}%"
            
            elif len(digits_only) == 3:
                # Remove last digit (as per your rule)
                battery_value = int(digits_only[:2])
                if 10 <= battery_value <= 99:
                    return f"{battery_value}%"
        
        return None
    
    def extract_temperature(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """🔧 FIXED: Extract temperature with proper handling of ~SC, ~IC, and OCR misreads.
        Rules:
        - ~SC = -5°C
        - S = 5 (e.g. 2SC = 25°C)
        - l, i, I = 1 (e.g. LC = 1°C, ~IC = -1°C)
        - Standard °C/°F patterns
        """
        for text in texts:
            # Clean the text first
            cleaned_text = text.strip().upper()

            # --- Handle special OCR misreads ---
            # Replace l, i, I with 1
            cleaned_text = cleaned_text.replace("L", "1").replace("I", "1")

            # --- Handle ~SC pattern specifically (highest priority) ---
            if "~SC" in cleaned_text:
                return {
                    "original_value": -5,
                    "original_unit": "C",
                    "celsius": -5,
                    "original": "-5°C",
                }

            # --- Handle ~XC patterns where X is a digit ---
            tilde_match = re.search(r"~(\d{1,2})\s*[Cc]", cleaned_text, re.IGNORECASE)
            if tilde_match:
                try:
                    temp_value = -int(tilde_match.group(1))  # Make negative
                    if -40 <= temp_value <= 10:  # Reasonable negative Celsius range
                        return {
                            "original_value": temp_value,
                            "original_unit": "C",
                            "celsius": temp_value,
                            "original": f"{temp_value}°C",
                        }
                except ValueError:
                    pass

            # --- Handle ~XF patterns ---
            tilde_f_match = re.search(r"~(\d{1,2})\s*[Ff]", cleaned_text, re.IGNORECASE)
            if tilde_f_match:
                try:
                    temp_value = -int(tilde_f_match.group(1))  # Make negative
                    if -40 <= temp_value <= 50:  # Reasonable negative Fahrenheit range
                        celsius = round((temp_value - 32) * 5 / 9, 1)
                        return {
                            "original_value": temp_value,
                            "original_unit": "F",
                            "celsius": celsius,
                            "original": f"{temp_value}°F",
                        }
                except ValueError:
                    pass

            # --- Handle XSC pattern where S = 5 ---
            sc_match = re.search(r"(\d{1,2})[Ss][Cc]", cleaned_text, re.IGNORECASE)
            if sc_match:
                try:
                    first_digit = int(sc_match.group(1))
                    temp_value = int(f"{first_digit}5")  # S = 5
                    if -40 <= temp_value <= 60:
                        return {
                            "original_value": temp_value,
                            "original_unit": "C",
                            "celsius": temp_value,
                            "original": f"{temp_value}°C",
                        }
                except ValueError:
                    pass

            # --- Handle 1C / 1F cases (from l/i/I replacement) ---
            one_match = re.search(r"^1\s*([CcFf])$", cleaned_text, re.IGNORECASE)
            if one_match:
                unit = one_match.group(1).upper()
                if unit == "C":
                    return {
                        "original_value": 1,
                        "original_unit": "C",
                        "celsius": 1,
                        "original": "1°C",
                    }
                elif unit == "F":
                    celsius = round((1 - 32) * 5 / 9, 1)
                    return {
                        "original_value": 1,
                        "original_unit": "F",
                        "celsius": celsius,
                        "original": "1°F",
                    }

            # --- Standard temperature patterns ---
            temp_match = re.search(r"(\d{1,3})\s*[°]?([CcFf])", cleaned_text, re.IGNORECASE)
            if temp_match:
                try:
                    temp_value = int(temp_match.group(1))
                    unit = temp_match.group(2).upper()

                    if unit == "F" and -40 <= temp_value <= 140:
                        celsius = round((temp_value - 32) * 5 / 9, 1)
                        return {
                            "original_value": temp_value,
                            "original_unit": "F",
                            "celsius": celsius,
                            "original": f"{temp_value}°F",
                        }
                    elif unit == "C" and -40 <= temp_value <= 60:
                        return {
                            "original_value": temp_value,
                            "original_unit": "C",
                            "celsius": temp_value,
                            "original": f"{temp_value}°C",
                        }
                except (ValueError, IndexError):
                    continue

        return None
    
    def _validate_date_components(self, year: int, month: int, day: int) -> bool:
        """Validate date components"""
        if not (1990 <= year <= 2030):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        
        # Check month-specific day limits
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day > days_in_month[month - 1]:
            if month == 2 and day == 29:
                # Check leap year
                return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            return False
        
        return True

class TimelapseAnalyzer:
    """Enhanced analyzer with consistent time prediction and sequence reset detection"""
    
    def __init__(self):
        self.sequences = defaultdict(list)
        self.detected_intervals = {}
        self.sequence_groups = defaultdict(lambda: defaultdict(list))
        self.cached_intervals = {}  
        # 🆕 Store predictions by sequence: {(date, seq_id): {seq_num: time}}
        self.last_predictions = defaultdict(dict)
            
    def _store_prediction(self, date_key: str, seq_id: str, seq_num: int, predicted_time: str):
        """Store prediction as 'virtual OCR' for future reference"""
        prediction_key = (date_key, seq_id)
        if prediction_key not in self.last_predictions:
            self.last_predictions[prediction_key] = {}
        
        self.last_predictions[prediction_key][seq_num] = predicted_time
        
    def _get_most_recent_reference_with_predictions(self, date_key: str, seq_id: str, target_seq: int) -> Optional[Tuple[int, str, int]]:
        """
        🎯 Find most recent reference - either OCR time OR stored prediction
        """
        images = self.sequence_groups[date_key][seq_id]
        
        # Get ALL images before target sequence, sorted by sequence number (descending)
        candidates = []
        for img in images:
            if img['sequence_number'] < target_seq:  # Only previous images
                candidates.append(img)
        
        if not candidates:
            return None
        
        # Sort by sequence number (descending) to get most recent first
        candidates.sort(key=lambda x: x['sequence_number'], reverse=True)
        
        interval_sec = self._get_stable_interval(date_key, seq_id)
        
        # 🎯 PRIORITY 1: Check for stored predictions (more recent)
        prediction_key = (date_key, seq_id)
        if prediction_key in self.last_predictions:
            stored_predictions = self.last_predictions[prediction_key]
            # Find most recent prediction before target
            for pred_seq in sorted(stored_predictions.keys(), reverse=True):
                if pred_seq < target_seq:
                    pred_time = stored_predictions[pred_seq]
                    return (pred_seq, pred_time, interval_sec)
        
        # 🎯 PRIORITY 2: Find most recent OCR time
        for img in candidates:
            if img.get('ocr_time'):
                return (img['sequence_number'], img['ocr_time'], interval_sec)
        
        return None

    def add_image_data(self, filename: str, filename_info: Dict, ocr_time: Optional[str], 
                      ocr_date: Optional[str]):
        """Add image data for timelapse analysis with sequence grouping"""
        if not filename_info.get('has_timelapse') or filename_info.get('sequence_number') is None:
            return
        
        date_key = filename_info['filename_date']
        if not date_key:
            return
            
        seq_id = filename_info.get('sequence_identifier', 'default')
        
        image_data = {
            'filename': filename,
            'sequence_number': filename_info['sequence_number'],
            'sequence_identifier': seq_id,
            'frame_id': filename_info.get('frame_id'),
            'ocr_time': ocr_time,
            'ocr_date': ocr_date,
            'filename_info': filename_info,
            'date_key': date_key
        }
        
        # Add to both structures
        self.sequences[date_key].append(image_data)
        self.sequence_groups[date_key][seq_id].append(image_data)
    
    def _get_stable_interval(self, date_key: str, seq_id: str) -> int:
        """🔧 FIXED: Get stable interval that doesn't jump around for same sequence"""
        cache_key = (date_key, seq_id)
        
        # Return cached interval if we have it
        if cache_key in self.cached_intervals:
            return self.cached_intervals[cache_key]
        
        images = self.sequence_groups[date_key][seq_id]
        
        # Get ALL images with OCR times, sorted by sequence
        timed_images = []
        for img in images:
            if img.get('ocr_time'):
                timed_images.append((img['sequence_number'], img['ocr_time']))
        
        if len(timed_images) < 2:
            # Cache default interval
            self.cached_intervals[cache_key] = 10
            return 10
        
        timed_images.sort()
        
        # Calculate intervals from all pairs
        intervals = []
        for i in range(len(timed_images) - 1):
            seq1, time1 = timed_images[i]
            seq2, time2 = timed_images[i + 1]
            
            try:
                t1 = datetime.strptime(time1, "%H:%M:%S")
                t2 = datetime.strptime(time2, "%H:%M:%S")
                
                # Handle day rollover
                if t2 < t1:
                    t2 += timedelta(days=1)
                
                time_diff_sec = (t2 - t1).total_seconds()
                seq_diff = seq2 - seq1
                
                if seq_diff > 0:
                    interval = time_diff_sec / seq_diff
                    if 1 <= interval <= 600:  # 1 second to 10 minutes
                        intervals.append(int(round(interval)))
            except ValueError:
                continue
        
        if intervals:
            # Use most common interval (mode)
            from collections import Counter
            most_common_interval = Counter(intervals).most_common(1)[0][0]
            
            # Cache it
            self.cached_intervals[cache_key] = most_common_interval
            return most_common_interval
        
        # Cache default fallback
        self.cached_intervals[cache_key] = 10
        return 10

    def get_live_prediction(self, date_key: str, seq_id: str, seq_num: int) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        🎯 Use most recent reference (OCR or prediction), build chain of predictions
        """
        if date_key not in self.sequence_groups or seq_id not in self.sequence_groups[date_key]:
            return None, None, None
            
        images = self.sequence_groups[date_key][seq_id]
        if len(images) < 1:
            return None, None, None
        
        # Sort by sequence number
        images.sort(key=lambda x: x['sequence_number'])
        
        # 🆕 FIXED: Find most recent reference (OCR OR prediction)
        most_recent_ref = self._get_most_recent_reference_with_predictions(date_key, seq_id, seq_num)
        
        if most_recent_ref:
            ref_seq, ref_time, interval_sec = most_recent_ref
            
            try:
                ref_time_obj = datetime.strptime(ref_time, "%H:%M:%S")
                seq_diff = seq_num - ref_seq
                predicted_time_obj = ref_time_obj + timedelta(seconds=seq_diff * interval_sec)
                
                # Handle day rollover properly
                if predicted_time_obj.day != ref_time_obj.day:
                    # Wrap around to same day but preserve the time progression
                    predicted_time_obj = predicted_time_obj.replace(day=ref_time_obj.day)
                    if predicted_time_obj < ref_time_obj:
                        # This means we've wrapped around midnight
                        predicted_time_obj += timedelta(days=1)
                    predicted_time_obj = predicted_time_obj.replace(day=ref_time_obj.day)
                
                predicted_time = predicted_time_obj.strftime("%H:%M:%S")
                
                # 🆕 Store this prediction as "virtual OCR" for future use
                self._store_prediction(date_key, seq_id, seq_num, predicted_time)
                
                confidence = self._calculate_confidence(date_key, seq_id)
                samples = len([img for img in images if img.get('ocr_time')])
                
                return interval_sec, predicted_time, f"{confidence:.1%} confidence, {samples} samples"
                
            except ValueError:
                pass
        
        return None, None, "No reference found"

    def _get_most_recent_reference(self, date_key: str, seq_id: str, target_seq: int) -> Optional[Tuple[int, str, int]]:
        """
        🎯 ALWAYS get the most recent reference with STABLE interval
        """
        images = self.sequence_groups[date_key][seq_id]
        
        # Get ALL images before target sequence, sorted by sequence number (descending)
        candidates = []
        for img in images:
            if img['sequence_number'] < target_seq:  # Only previous images
                candidates.append(img)
        
        if not candidates:
            return None
        
        # Sort by sequence number (descending) to get most recent first
        candidates.sort(key=lambda x: x['sequence_number'], reverse=True)
        
        # 🎯 PRIORITY 1: Find most recent OCR time
        for img in candidates:
            if img.get('ocr_time'):
                # Use STABLE interval calculation
                interval_sec = self._get_stable_interval(date_key, seq_id)
                return (img['sequence_number'], img['ocr_time'], interval_sec)
        
        return None

    def _get_fast_interval(self, date_key: str, seq_id: str) -> int:
        """🚀 Get interval from most recent OCR pairs only"""
        images = self.sequence_groups[date_key][seq_id]
        
        # Get ALL images with OCR times, sorted by sequence
        timed_images = []
        for img in images:
            if img.get('ocr_time'):
                timed_images.append((img['sequence_number'], img['ocr_time']))
        
        if len(timed_images) < 2:
            return 10  # Default 10 second interval
        
        timed_images.sort()
        
        # 🎯 Use ONLY the most recent 2 OCR pairs for interval
        if len(timed_images) >= 2:
            # Take the last pair for most accurate current interval
            seq1, time1 = timed_images[-2]
            seq2, time2 = timed_images[-1]
            
            try:
                t1 = datetime.strptime(time1, "%H:%M:%S")
                t2 = datetime.strptime(time2, "%H:%M:%S")
                
                # Handle day rollover
                if t2 < t1:
                    t2 += timedelta(days=1)
                
                time_diff_sec = (t2 - t1).total_seconds()
                seq_diff = seq2 - seq1
                
                if seq_diff > 0:
                    interval = time_diff_sec / seq_diff
                    if 1 <= interval <= 600:  # Reasonable range
                        return int(round(interval))
            except ValueError:
                pass
        
        return 10  # Default fallback
    
    def _update_interval_calculation(self, date_key: str, seq_id: str, seq_num: int, ocr_time: str) -> int:
        """Update interval calculation with new OCR time data point"""
        images = self.sequence_groups[date_key][seq_id]
        
        # Find all timed images up to current sequence (sorted)
        timed_images = []
        for img in images:
            if img['sequence_number'] <= seq_num and img.get('ocr_time'):
                timed_images.append((img['sequence_number'], img['ocr_time']))
        
        timed_images.sort()
        
        if len(timed_images) < 2:
            return 10  # Default 10 second interval
        
        # Calculate intervals from recent samples (last 10 pairs max)
        intervals = []
        start_idx = max(0, len(timed_images) - 10)
        
        for i in range(start_idx, len(timed_images) - 1):
            seq1, time1 = timed_images[i]
            seq2, time2 = timed_images[i + 1]
            
            try:
                t1 = datetime.strptime(time1, "%H:%M:%S")
                t2 = datetime.strptime(time2, "%H:%M:%S")
                
                # Handle day rollover
                if t2 < t1:
                    t2 += timedelta(days=1)
                
                time_diff_sec = (t2 - t1).total_seconds()
                seq_diff = seq2 - seq1
                
                if seq_diff > 0:
                    interval = time_diff_sec / seq_diff
                    if 1 <= interval <= 600:  # 1 second to 10 minutes reasonable
                        intervals.append(int(round(interval)))
            except ValueError:
                continue
        
        if intervals:
            # Use most common recent interval
            return Counter(intervals).most_common(1)[0][0]
        
        return 10  # Default fallback
    
    def _calculate_initial_prediction(self, date_key: str, seq_id: str, seq_num: int) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Calculate initial prediction when no reference exists"""
        images = self.sequence_groups[date_key][seq_id]
        
        # Find images with OCR times
        timed_images = [(img['sequence_number'], img['ocr_time']) 
                       for img in images if img.get('ocr_time')]
        
        if len(timed_images) < 1:
            return None, None, None
        
        timed_images.sort()
        
        # If only one timed image, use it as reference with default interval
        if len(timed_images) == 1:
            ref_seq, ref_time = timed_images[0]
            default_interval = 10
            
            try:
                ref_time_obj = datetime.strptime(ref_time, "%H:%M:%S")
                seq_diff = seq_num - ref_seq
                predicted_time_obj = ref_time_obj + timedelta(seconds=seq_diff * default_interval)
                predicted_time = predicted_time_obj.strftime("%H:%M:%S")
                
                # Store as reference
                prediction_key = (date_key, seq_id)
                self.last_predictions[prediction_key] = (seq_num, predicted_time, default_interval)
                
                return default_interval, predicted_time, "100.0% confidence (single reference)"
                
            except ValueError:
                return None, None, None
        
        # Calculate intervals from multiple timed images
        intervals = []
        for i in range(len(timed_images) - 1):
            seq1, time1 = timed_images[i]
            seq2, time2 = timed_images[i + 1]
            
            try:
                t1 = datetime.strptime(time1, "%H:%M:%S")
                t2 = datetime.strptime(time2, "%H:%M:%S")
                
                # Handle day rollover
                if t2 < t1:
                    t2 += timedelta(days=1)
                
                time_diff_sec = (t2 - t1).total_seconds()
                seq_diff = seq2 - seq1
                
                if seq_diff > 0:
                    interval = time_diff_sec / seq_diff
                    if 1 <= interval <= 600:
                        intervals.append(int(round(interval)))
            except ValueError:
                continue
        
        if not intervals:
            return None, None, None
        
        best_interval = Counter(intervals).most_common(1)[0][0]
        
        # Find best reference point (closest before or at target sequence)
        ref_seq, ref_time = None, None
        for seq, time in reversed(timed_images):
            if seq <= seq_num:
                ref_seq, ref_time = seq, time
                break
        
        # If no reference before target, use closest after
        if ref_seq is None:
            min_distance = float('inf')
            for seq, time in timed_images:
                distance = abs(seq - seq_num)
                if distance < min_distance:
                    min_distance = distance
                    ref_seq, ref_time = seq, time
        
        if ref_seq is None:
            return None, None, None
        
        try:
            ref_time_obj = datetime.strptime(ref_time, "%H:%M:%S")
            seq_diff = seq_num - ref_seq
            predicted_time_obj = ref_time_obj + timedelta(seconds=seq_diff * best_interval)
            predicted_time = predicted_time_obj.strftime("%H:%M:%S")
            
            # Store as reference
            prediction_key = (date_key, seq_id)
            self.last_predictions[prediction_key] = (seq_num, predicted_time, best_interval)
            
            confidence = len(intervals) / max(1, len(timed_images) - 1)
            return best_interval, predicted_time, f"{confidence:.1%} confidence (initial)"
            
        except ValueError:
            return best_interval, None, "time calculation error"
    
    def _calculate_confidence(self, date_key: str, seq_id: str) -> float:
        """Calculate prediction confidence based on OCR time availability"""
        images = self.sequence_groups[date_key][seq_id]
        timed_count = sum(1 for img in images if img.get('ocr_time'))
        total_count = len(images)
        
        if total_count == 0:
            return 0.0
        
        # Confidence based on ratio of timed images
        base_confidence = timed_count / total_count
        
        # Boost confidence if we have enough samples
        if timed_count >= 10:
            base_confidence = min(1.0, base_confidence * 1.2)
        elif timed_count >= 5:
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence
    
    def detect_sequence_resets(self, date_key: str, seq_id: str) -> List[Dict]:
        """Detect sequence number resets within the same day/sequence"""
        if date_key not in self.sequence_groups or seq_id not in self.sequence_groups[date_key]:
            return []
        
        images = self.sequence_groups[date_key][seq_id]
        images.sort(key=lambda x: x['sequence_number'])
        
        resets = []
        for i in range(1, len(images)):
            prev_seq = images[i-1]['sequence_number']
            curr_seq = images[i]['sequence_number']
            prev_time = images[i-1].get('ocr_time')
            curr_time = images[i].get('ocr_time')
            
            # Detect reset: significant sequence drop with time continuation
            if curr_seq < prev_seq - 100 and prev_time and curr_time:  # At least 100 sequence drop
                try:
                    prev_t = datetime.strptime(prev_time, "%H:%M:%S")
                    curr_t = datetime.strptime(curr_time, "%H:%M:%S")
                    
                    # If time continues or goes back less than 2 hours, likely a reset
                    time_diff = (prev_t - curr_t).total_seconds()
                    if curr_t >= prev_t or time_diff < 7200:  # Less than 2 hours backwards
                        resets.append({
                            'reset_at_index': i,
                            'prev_seq': prev_seq,
                            'curr_seq': curr_seq,
                            'prev_time': prev_time,
                            'curr_time': curr_time,
                            'estimated_continue_seq': prev_seq + 1,
                            'time_diff_seconds': time_diff
                        })
                except ValueError:
                    pass
        
        return resets
    
    def get_current_prediction(self, date_key: str, seq_num: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Legacy method for compatibility - uses default sequence ID
        """
        seq_id = 'default'
        interval, predicted_time, _ = self.get_live_prediction(date_key, seq_id, seq_num)
        return interval, predicted_time
    
    def build_global_sequence_index(self):
        """
        Build a monotonic global sequence index across days, handling resets.
        """
        all_images = []
        for date_key, images in self.sequences.items():
            for img in images:
                all_images.append((date_key, img['sequence_number'], img))

        # Sort by date first, then by sequence number
        all_images.sort(key=lambda x: (x[0], x[1]))

        global_index = 0
        last_date = None
        last_seq = None
        day_offset = 0

        for date_key, seq_num, img in all_images:
            if last_date and date_key > last_date:
                # New day
                if seq_num < last_seq:
                    # Reset detected → continue timeline
                    day_offset += 1

            # Assign a global index that is monotonic across days
            img['global_index'] = global_index
            img['day_offset'] = day_offset

            global_index += 1
            last_date = date_key
            last_seq = seq_num

    def analyze_sequences(self) -> Dict[str, Any]:
        """Enhanced sequence analysis with reset detection"""
        results = {
            'detected_intervals': {},
            'sequence_stats': {},
            'sequence_resets': {},
            'anomalies': []
        }
        
        # Analyze each date/sequence combination
        for date_key, seq_groups in self.sequence_groups.items():
            results['detected_intervals'][date_key] = {}
            results['sequence_resets'][date_key] = {}
            
            for seq_id, images in seq_groups.items():
                if len(images) < 2:
                    continue
                
                # Sort by sequence number
                images.sort(key=lambda x: x['sequence_number'])
                
                # Detect resets
                resets = self.detect_sequence_resets(date_key, seq_id)
                if resets:
                    results['sequence_resets'][date_key][seq_id] = resets
                
                # Analyze intervals
                timed_images = [img for img in images if img.get('ocr_time')]
                if len(timed_images) >= 2:
                    interval_data = self._analyze_interval_for_sequence(timed_images)
                    if interval_data:
                        results['detected_intervals'][date_key][seq_id] = interval_data
        
        # Build sequence stats for compatibility
        for date_key, images in self.sequences.items():
            images.sort(key=lambda x: x['sequence_number'])
            timed_images = [img for img in images if img.get('ocr_time')]
            
            results['sequence_stats'][date_key] = {
                'total_images': len(images),
                'images_with_ocr_time': len(timed_images),
                'sequence_range': (min(img['sequence_number'] for img in images),
                                 max(img['sequence_number'] for img in images)),
                'missing_sequences': self._find_missing_sequences(images),
                'sequence_identifiers': list(set(img.get('sequence_identifier', 'default') for img in images))
            }
        
        # Build global sequence index
        self.build_global_sequence_index()
        
        return results
    
    def _analyze_interval_for_sequence(self, timed_images: List[Dict]) -> Optional[Dict]:
        """Analyze time intervals for a specific sequence"""
        if len(timed_images) < 2:
            return None
        
        intervals = []
        for i in range(len(timed_images) - 1):
            img1 = timed_images[i]
            img2 = timed_images[i + 1]
            
            try:
                time1 = datetime.strptime(img1['ocr_time'], '%H:%M:%S')
                time2 = datetime.strptime(img2['ocr_time'], '%H:%M:%S')
                
                # Handle day rollover
                if time2 < time1:
                    time2 += timedelta(days=1)
                
                time_diff = time2 - time1
                seq_diff = img2['sequence_number'] - img1['sequence_number']
                
                if seq_diff > 0:
                    interval_seconds = time_diff.total_seconds() / seq_diff
                    if 1 <= interval_seconds <= 600:  # 1 second to 10 minutes
                        intervals.append(interval_seconds)
            except ValueError:
                continue
        
        if not intervals:
            return None
        
        interval_values = [int(round(i)) for i in intervals]
        interval_counter = Counter(interval_values)
        most_common_interval = interval_counter.most_common(1)[0][0]
        
        return {
            'interval_seconds': most_common_interval,
            'confidence': interval_counter[most_common_interval] / len(intervals),
            'total_samples': len(intervals),
            'all_intervals': intervals,
            'predicted_times': self._generate_predicted_times_for_sequence(timed_images, most_common_interval)
        }
    
    def _generate_predicted_times_for_sequence(self, timed_images: List[Dict], interval_seconds: int) -> Dict[int, str]:
        """Generate predicted times for all sequence numbers in the range"""
        if not timed_images:
            return {}
        
        # Use the first timed image as reference
        reference = timed_images[0]
        ref_time = datetime.strptime(reference['ocr_time'], '%H:%M:%S')
        ref_seq = reference['sequence_number']
        
        # Get sequence range from all images
        all_seqs = [img['sequence_number'] for img in timed_images]
        min_seq = min(all_seqs)
        max_seq = max(all_seqs)
        
        predictions = {}
        for seq_num in range(min_seq, max_seq + 1):
            seq_diff = seq_num - ref_seq
            predicted_time = ref_time + timedelta(seconds=seq_diff * interval_seconds)
            predictions[seq_num] = predicted_time.strftime('%H:%M:%S')
        
        return predictions

    def _find_missing_sequences(self, images: List[Dict]) -> List[int]:
        """Find missing sequence numbers in the range"""
        if not images:
            return []
        
        seq_numbers = [img['sequence_number'] for img in images]
        min_seq = min(seq_numbers)
        max_seq = max(seq_numbers)
        
        expected = set(range(min_seq, max_seq + 1))
        actual = set(seq_numbers)
        
        return sorted(list(expected - actual))

class PostProcessor:
    """🆕 Advanced post-processing for temperature, battery, and time correction"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.corrections_made = {
            'temperature_filled': 0,
            'temperature_corrected': 0,
            'battery_filled': 0,
            'battery_corrected': 0,
            'time_filled': 0,
            'time_corrected': 0
        }
    
    def process_sequences(self, results: List[Dict], timelapse_analysis: Dict) -> List[Dict]:
        """Process all sequences with advanced corrections"""
        
        # Group by date and sequence identifier
        grouped_results = defaultdict(list)
        
        for result in results:
            filename_info = result.get('filename_info', {})
            date_key = filename_info.get('filename_date')
            seq_id = filename_info.get('sequence_identifier', 'default')
            
            if date_key and filename_info.get('has_timelapse'):
                grouped_results[f"{date_key}_{seq_id}"].append(result)
        
        # Process each group
        processed_results = []
        for group_key, group_results in grouped_results.items():
            date_key = group_key.split('_')[0]
            
            # Sort by sequence number
            group_results.sort(key=lambda x: x.get('filename_info', {}).get('sequence_number', 0))
            
            # Apply corrections
            group_results = self._correct_temperatures(group_results)
            group_results = self._correct_battery_levels(group_results)
            group_results = self._correct_times(group_results, timelapse_analysis.get('detected_intervals', {}).get(date_key))
            
            processed_results.extend(group_results)
        
        # Add non-timelapse results unchanged
        for result in results:
            filename_info = result.get('filename_info', {})
            if not filename_info.get('has_timelapse'):
                processed_results.append(result)
        
        return processed_results
        
    def _correct_temperatures(self, sequence_results: List[Dict]) -> List[Dict]:
        """🌡️ Correct temperature anomalies with neighbour, global, and rate-of-change checks"""

        # Extract temperature data
        temp_data = []
        for i, result in enumerate(sequence_results):
            temp_c = result.get('Temperature_C')
            seq_num = result.get('filename_info', {}).get('sequence_number', i)
            temp_data.append({
                'index': i,
                'seq_num': seq_num,
                'temp': temp_c,
                'original_temp': temp_c
            })

        if len(temp_data) < 3:
            return sequence_results

        # Step 1: Neighbourhood outlier detection
        for i in range(1, len(temp_data) - 1):
            curr = temp_data[i]
            prev_temp = temp_data[i-1]['temp']
            next_temp = temp_data[i+1]['temp']
            curr_temp = curr['temp']

            if curr_temp is not None and prev_temp is not None and next_temp is not None:
                if abs(prev_temp - next_temp) <= 2 and (abs(curr_temp - prev_temp) > 15 and abs(curr_temp - next_temp) > 15):
                    # Outlier → discard
                    sequence_results[i]['Temperature_C'] = None
                    sequence_results[i]['temperature_corrected'] = True
                    sequence_results[i]['temperature_correction_reason'] = f"Neighbour outlier: {curr_temp}°C discarded"
                    temp_data[i]['temp'] = None
                    self.corrections_made['temperature_corrected'] += 1
                    if self.debug:
                        print(f"🌡️ NEIGHBOUR OUTLIER REMOVED: Seq {curr['seq_num']} {curr_temp}°C → None")

        # Step 2: Global distribution check (only if enough samples)
        valid_temps = [t['temp'] for t in temp_data if t['temp'] is not None]
        if len(valid_temps) > 100:
            sorted_temps = sorted(valid_temps)
            second_highest = sorted_temps[-2] if len(sorted_temps) >= 2 else sorted_temps[-1]
            highest = sorted_temps[-1]
            second_lowest = sorted_temps[1] if len(sorted_temps) >= 2 else sorted_temps[0]
            lowest = sorted_temps[0]

            for i, entry in enumerate(temp_data):
                t = entry['temp']
                if t is None:
                    continue

                # If max is way above second highest (e.g. 37 vs 28)
                if t == highest and highest - second_highest > 5:
                    sequence_results[i]['Temperature_C'] = None
                    sequence_results[i]['temperature_corrected'] = True
                    sequence_results[i]['temperature_correction_reason'] = f"Global outlier high: {t}°C discarded"
                    temp_data[i]['temp'] = None
                    self.corrections_made['temperature_corrected'] += 1
                    if self.debug:
                        print(f"🌡️ GLOBAL HIGH OUTLIER REMOVED: Seq {entry['seq_num']} {t}°C → None")

                # If min is way below second lowest
                if t == lowest and second_lowest - lowest > 5:
                    sequence_results[i]['Temperature_C'] = None
                    sequence_results[i]['temperature_corrected'] = True
                    sequence_results[i]['temperature_correction_reason'] = f"Global outlier low: {t}°C discarded"
                    temp_data[i]['temp'] = None
                    self.corrections_made['temperature_corrected'] += 1
                    if self.debug:
                        print(f"🌡️ GLOBAL LOW OUTLIER REMOVED: Seq {entry['seq_num']} {t}°C → None")

        # Step 3: Rate-of-change check
        for i in range(1, len(temp_data)):
            curr = temp_data[i]
            prev = temp_data[i-1]
            if curr['temp'] is not None and prev['temp'] is not None:
                seq_diff = curr['seq_num'] - prev['seq_num']
                if seq_diff <= 0:
                    continue
                temp_diff = abs(curr['temp'] - prev['temp'])
                # If jump > 10°C within 1 hour (360 frames at 10s interval), discard
                if seq_diff <= 360 and temp_diff > 10:
                    sequence_results[i]['Temperature_C'] = None
                    sequence_results[i]['temperature_corrected'] = True
                    sequence_results[i]['temperature_correction_reason'] = f"Rate-of-change outlier: {curr['temp']}°C discarded"
                    temp_data[i]['temp'] = None
                    self.corrections_made['temperature_corrected'] += 1
                    if self.debug:
                        print(f"🌡️ RATE-OF-CHANGE OUTLIER REMOVED: Seq {curr['seq_num']} {curr['temp']}°C → None")

        # Step 4: Fill gaps using interpolation
        for i in range(len(temp_data)):
            if temp_data[i]['temp'] is None:
                left_temp, right_temp = None, None
                left_idx, right_idx = i-1, i+1

                while left_idx >= 0 and temp_data[left_idx]['temp'] is None:
                    left_idx -= 1
                if left_idx >= 0:
                    left_temp = temp_data[left_idx]['temp']

                while right_idx < len(temp_data) and temp_data[right_idx]['temp'] is None:
                    right_idx += 1
                if right_idx < len(temp_data):
                    right_temp = temp_data[right_idx]['temp']

                if left_temp is not None and right_temp is not None:
                    interpolated = round((left_temp + right_temp) / 2)
                    temp_data[i]['temp'] = interpolated
                    sequence_results[i]['Temperature_C'] = interpolated
                    sequence_results[i]['temperature_filled'] = True
                    sequence_results[i]['temperature_fill_reason'] = f"Interpolated between {left_temp}°C and {right_temp}°C"
                    self.corrections_made['temperature_filled'] += 1
                elif left_temp is not None:
                    temp_data[i]['temp'] = left_temp
                    sequence_results[i]['Temperature_C'] = left_temp
                    sequence_results[i]['temperature_filled'] = True
                    sequence_results[i]['temperature_fill_reason'] = f"Forward fill from {left_temp}°C"
                    self.corrections_made['temperature_filled'] += 1
                elif right_temp is not None:
                    temp_data[i]['temp'] = right_temp
                    sequence_results[i]['Temperature_C'] = right_temp
                    sequence_results[i]['temperature_filled'] = True
                    sequence_results[i]['temperature_fill_reason'] = f"Backward fill from {right_temp}°C"
                    self.corrections_made['temperature_filled'] += 1

        return sequence_results
    
    def _correct_battery_levels(self, sequence_results: List[Dict]) -> List[Dict]:
        """🔋 Correct battery level anomalies (max 1% drop per 100 frames)"""
        
        # Extract battery data
        battery_data = []
        for i, result in enumerate(sequence_results):
            battery_str = result.get('Battery_Level', '')
            battery_val = None
            if battery_str and battery_str.endswith('%'):
                try:
                    battery_val = int(battery_str[:-1])
                except ValueError:
                    pass
            
            seq_num = result.get('filename_info', {}).get('sequence_number', i)
            battery_data.append({
                'index': i,
                'seq_num': seq_num,
                'battery': battery_val,
                'original_battery': battery_val
            })
        
        if len(battery_data) < 3:
            return sequence_results
        
        # Step 1: Correct impossible jumps
        for i in range(1, len(battery_data)):
            curr = battery_data[i]
            prev = battery_data[i-1]
            
            if curr['battery'] is not None and prev['battery'] is not None:
                seq_diff = curr['seq_num'] - prev['seq_num']
                battery_diff = curr['battery'] - prev['battery']
                
                # Battery can only drop, max 1% per 100 frames
                max_drop = max(1, seq_diff // 100)
                
                if battery_diff > 0:  # Battery increased (impossible)
                    corrected_battery = prev['battery']
                    battery_data[i]['battery'] = corrected_battery
                    sequence_results[i]['Battery_Level'] = f"{corrected_battery}%"
                    sequence_results[i]['battery_corrected'] = True
                    sequence_results[i]['battery_correction_reason'] = f"Impossible increase: {curr['original_battery']}% → {corrected_battery}%"
                    self.corrections_made['battery_corrected'] += 1
                    
                elif abs(battery_diff) > max_drop:  # Too big drop
                    corrected_battery = prev['battery'] - max_drop
                    battery_data[i]['battery'] = corrected_battery
                    sequence_results[i]['Battery_Level'] = f"{corrected_battery}%"
                    sequence_results[i]['battery_corrected'] = True
                    sequence_results[i]['battery_correction_reason'] = f"Excessive drop: {curr['original_battery']}% → {corrected_battery}%"
                    self.corrections_made['battery_corrected'] += 1
        
        # Step 2: Fill gaps with gradual decline
        for i in range(len(battery_data)):
            if battery_data[i]['battery'] is None:
                # Find nearest valid batteries
                left_battery, right_battery = None, None
                left_idx, right_idx = i-1, i+1
                
                # Search left
                while left_idx >= 0 and battery_data[left_idx]['battery'] is None:
                    left_idx -= 1
                if left_idx >= 0:
                    left_battery = battery_data[left_idx]['battery']
                
                # Search right
                while right_idx < len(battery_data) and battery_data[right_idx]['battery'] is None:
                    right_idx += 1
                if right_idx < len(battery_data):
                    right_battery = battery_data[right_idx]['battery']
                
                # Fill gap
                if left_battery is not None and right_battery is not None:
                    # Linear interpolation with gradual decline
                    gap_size = right_idx - left_idx
                    position = i - left_idx
                    interpolated = left_battery + (right_battery - left_battery) * (position / gap_size)
                    filled_battery = max(0, round(interpolated))
                    
                    battery_data[i]['battery'] = filled_battery
                    sequence_results[i]['Battery_Level'] = f"{filled_battery}%"
                    sequence_results[i]['battery_filled'] = True
                    self.corrections_made['battery_filled'] += 1
                    
                elif left_battery is not None:
                    # Gradual decline from left
                    seq_diff = battery_data[i]['seq_num'] - battery_data[left_idx]['seq_num']
                    decline = min(left_battery, seq_diff // 100)
                    filled_battery = max(0, left_battery - decline)
                    
                    battery_data[i]['battery'] = filled_battery
                    sequence_results[i]['Battery_Level'] = f"{filled_battery}%"
                    sequence_results[i]['battery_filled'] = True
                    self.corrections_made['battery_filled'] += 1
        
        return sequence_results
        
    def _correct_times(self, sequence_results: List[Dict], interval_data: Optional[Dict]) -> List[Dict]:
        """⏰ Enhanced time correction with sequence-aware interpolation"""
        
        if not sequence_results:
            return sequence_results
        
        # Group by sequence identifier for targeted processing
        seq_groups = defaultdict(list)
        for i, result in enumerate(sequence_results):
            filename_info = result.get('filename_info', {})
            seq_id = filename_info.get('sequence_identifier', 'default')
            seq_groups[seq_id].append((i, result))
        
        # Process each sequence group
        for seq_id, group in seq_groups.items():
            group.sort(key=lambda x: x[1].get('filename_info', {}).get('sequence_number', 0))
            
            # Extract time data with sequence numbers
            time_data = []
            for i, (result_idx, result) in enumerate(group):
                seq_num = result.get('filename_info', {}).get('sequence_number', 0)
                date_str = result.get('Date', '')
                
                ocr_time = None
                has_valid_time = False
                
                if date_str and ' ' in date_str:
                    date_part, time_part = date_str.split(' ', 1)
                    if time_part != "00:00:00":
                        ocr_time = time_part
                        has_valid_time = True
                
                time_data.append({
                    'result_idx': result_idx,
                    'seq_num': seq_num,
                    'ocr_time': ocr_time,
                    'has_valid_time': has_valid_time,
                    'date_part': date_str.split(' ')[0] if ' ' in date_str else date_str.split(' ')[0] if date_str else None
                })
            
            # Calculate intervals from valid OCR times
            interval_sec = self._calculate_sequence_interval(time_data)
            
            if interval_sec is None:
                continue
            
            # Fill gaps using interpolation
            self._fill_time_gaps(time_data, sequence_results, interval_sec)
        
        return sequence_results

    def _fill_time_gaps(self, time_data: List[Dict], sequence_results: List[Dict], interval_sec: int):
        """Fill time gaps using sequence-based interpolation"""
        
        # Find reference points (images with valid OCR times)
        reference_points = []
        for data in time_data:
            if data['has_valid_time']:
                reference_points.append((data['seq_num'], data['ocr_time']))
        
        if not reference_points:
            return
        
        # Fill gaps for each image
        for data in time_data:
            if data['has_valid_time']:
                continue  # Already has valid time
            
            result_idx = data['result_idx']
            target_seq = data['seq_num']
            date_part = data['date_part']
            
            if not date_part:
                continue
            
            # Find best reference point
            best_ref = self._find_closest_reference(reference_points, target_seq)
            if not best_ref:
                continue
            
            ref_seq, ref_time = best_ref
            
            try:
                ref_time_obj = datetime.strptime(ref_time, "%H:%M:%S")
                seq_diff = target_seq - ref_seq
                predicted_time_obj = ref_time_obj + timedelta(seconds=seq_diff * interval_sec)
                predicted_time = predicted_time_obj.strftime("%H:%M:%S")
                
                # Update the result
                sequence_results[result_idx]['Date'] = f"{date_part} {predicted_time}"
                sequence_results[result_idx]['time_filled'] = True
                sequence_results[result_idx]['time_fill_reason'] = f"Interpolated from seq {ref_seq} with {interval_sec}s interval"
                sequence_results[result_idx]['predicted_interval'] = interval_sec
                
                self.corrections_made['time_filled'] += 1
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"    🔧 TIME FILLED: Seq {target_seq} → {predicted_time} (ref: seq {ref_seq}, interval: {interval_sec}s)")
                    
            except ValueError:
                continue

    def _find_closest_reference(self, reference_points: List[Tuple[int, str]], target_seq: int) -> Optional[Tuple[int, str]]:
        """Find the closest reference point for time prediction"""
        if not reference_points:
            return None
        
        # Prefer reference before target sequence
        before_refs = [(seq, time) for seq, time in reference_points if seq <= target_seq]
        if before_refs:
            return max(before_refs, key=lambda x: x[0])  # Closest before
        
        # If no reference before, use closest after
        after_refs = [(seq, time) for seq, time in reference_points if seq > target_seq]
        if after_refs:
            return min(after_refs, key=lambda x: x[0])  # Closest after
        
        return None

    def _calculate_sequence_interval(self, time_data: List[Dict]) -> Optional[int]:
        """Calculate time interval for a sequence based on OCR times"""
        valid_times = [(data['seq_num'], data['ocr_time']) for data in time_data if data['has_valid_time']]
        
        if len(valid_times) < 2:
            return None
        
        intervals = []
        for i in range(len(valid_times) - 1):
            seq1, time1 = valid_times[i]
            seq2, time2 = valid_times[i + 1]
            
            try:
                t1 = datetime.strptime(time1, "%H:%M:%S")
                t2 = datetime.strptime(time2, "%H:%M:%S")
                
                # Handle day rollover
                if t2 < t1:
                    t2 += timedelta(days=1)
                
                time_diff_sec = (t2 - t1).total_seconds()
                seq_diff = seq2 - seq1
                
                if seq_diff > 0:
                    interval = time_diff_sec / seq_diff
                    if 1 <= interval <= 300:  # 1 second to 5 minutes
                        intervals.append(int(round(interval)))
            except ValueError:
                continue
        
        if not intervals:
            return None
        
        # Use most common interval
        return Counter(intervals).most_common(1)[0][0]    

    def print_correction_stats(self):
        """Print post-processing statistics"""
        print(f"\n🔧 POST-PROCESSING CORRECTIONS:")
        print(f"   🌡️  Temperature filled: {self.corrections_made['temperature_filled']:,}")
        print(f"   🌡️  Temperature corrected: {self.corrections_made['temperature_corrected']:,}")
        print(f"   🔋 Battery filled: {self.corrections_made['battery_filled']:,}")
        print(f"   🔋 Battery corrected: {self.corrections_made['battery_corrected']:,}")
        print(f"   ⏰ Time filled: {self.corrections_made['time_filled']:,}")
        print(f"   ⏰ Time corrected: {self.corrections_made['time_corrected']:,}")

class PlotwatcherExtractor:
    """Enhanced Plotwatcher extractor with timelapse analysis"""
    
    def __init__(self, ocr_engine: str = "easyocr", confidence: float = 0.3, 
                 debug: bool = False, max_workers: int = 1):
        self.ocr_processor = PlotwatcherOCR(engine=ocr_engine, confidence=confidence)
        self.pattern_matcher = PlotwatcherPatterns()
        self.timelapse_analyzer = TimelapseAnalyzer()
        self.post_processor = PostProcessor(debug=debug)
        self.debug = debug
        self.max_workers = max_workers
        
        # Enhanced statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'dates_extracted': 0,
            'times_extracted': 0,
            'temperatures_extracted': 0,
            'battery_levels_extracted': 0,
            'filename_dates_parsed': 0,
            'timelapse_sequences_detected': 0,
            'processing_time': 0.0,
            'filename_formats': Counter(),
            'temperature_units': Counter(),
            'sequence_identifiers': Counter(),
        }
        
        self.debug_failures = []
        
        logger.info(f"Enhanced PlotwatcherExtractor initialized with {ocr_engine.upper()}")
    
    def process_images(self, image_paths) -> List[Dict[str, Any]]:
        """🔧 ENHANCED: Process images with before/after comparison"""
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                image_paths = self._scan_directory(image_paths)
            else:
                image_paths = [image_paths]
        
        total_images = len(image_paths)
        logger.info(f"Processing {total_images} Plotwatcher images")
        start_time = time.time()
        
        results = []
        
        # Phase 1 - Process all images with progress tracking
        print(f"📸 Phase 1: Processing {total_images} images...")
        for i, path in enumerate(image_paths):
            result = self._process_single_image(path)
            results.append(result)
            
            # Add to timelapse analyzer
            filename_info = result.get('filename_info', {})
            ocr_time = None
            ocr_date = None
            
            if result.get('Date'):
                try:
                    dt_parts = result['Date'].split(' ')
                    if len(dt_parts) >= 2:
                        ocr_date = dt_parts[0]
                        ocr_time = dt_parts[1]
                except:
                    pass
            
            self.timelapse_analyzer.add_image_data(
                Path(path).name, filename_info, ocr_time, ocr_date
            )
            
            # Progress tracking
            current_count = i + 1
            if current_count % 100 == 0 or current_count == total_images:
                print(f"📸 Progress: {current_count}/{total_images} images processed")
        
        print("🔍 Phase 2: Analyzing timelapse sequences...")
        
        # Phase 2: Analyze timelapse sequences
        timelapse_analysis = self.timelapse_analyzer.analyze_sequences()
        
        print("🔧 Phase 3: Advanced post-processing...")
        
        # 🆕 BEFORE/AFTER COMPARISON
        if self.debug:
            print("\n" + "="*80)
            print("📊 BEFORE POST-PROCESSING - SAMPLE RESULTS")
            print("="*80)
            self._print_detailed_sample(results[:10], "BEFORE")
        
        # Phase 3: Advanced post-processing
        processed_results = self.post_processor.process_sequences(results, timelapse_analysis)
        
        if self.debug:
            print("\n" + "="*80)
            print("📊 AFTER POST-PROCESSING - SAMPLE RESULTS")
            print("="*80)
            self._print_detailed_sample(processed_results[:10], "AFTER")
            
            # Show correction summary
            print("\n" + "="*80)
            print("🔧 POST-PROCESSING CORRECTIONS SUMMARY")
            print("="*80)
            self._print_correction_summary(results, processed_results)
        
        processing_time = time.time() - start_time
        self.stats['total_processed'] = len(processed_results)
        self.stats['processing_time'] = processing_time
        self.stats['timelapse_sequences_detected'] = len(timelapse_analysis['detected_intervals'])
        
        # Store analysis for reporting
        self.timelapse_analysis = timelapse_analysis
        
        print(f"✅ Processing complete!")
        
        return processed_results
    
    def _print_correction_summary(self, before_results: List[Dict], after_results: List[Dict]):
        """Print summary of corrections made during post-processing"""
        
        # Count changes
        temp_filled = sum(1 for r in after_results if r.get('temperature_filled'))
        temp_corrected = sum(1 for r in after_results if r.get('temperature_corrected'))
        battery_filled = sum(1 for r in after_results if r.get('battery_filled'))
        battery_corrected = sum(1 for r in after_results if r.get('battery_corrected'))
        time_filled = sum(1 for r in after_results if r.get('time_filled'))
        time_corrected = sum(1 for r in after_results if r.get('time_corrected'))
        
        print(f"🌡️  Temperature Corrections:")
        print(f"   • Gaps filled: {temp_filled}")
        print(f"   • Outliers corrected: {temp_corrected}")
        
        print(f"\n🔋 Battery Corrections:")
        print(f"   • Gaps filled: {battery_filled}")
        print(f"   • Impossible values corrected: {battery_corrected}")
        
        print(f"\n⏰ Time Corrections:")
        print(f"   • Missing times filled: {time_filled}")
        print(f"   • Incorrect times corrected: {time_corrected}")
        
        # Show specific examples of corrections
        print(f"\n🔍 CORRECTION EXAMPLES:")
        
        correction_examples = []
        for i, (before, after) in enumerate(zip(before_results, after_results)):
            if len(correction_examples) >= 5:  # Limit to 5 examples
                break
                
            filename = before.get('filename', f'image_{i}')
            changes = []
            
            # Temperature changes
            before_temp = before.get('Temperature_C')
            after_temp = after.get('Temperature_C')
            if before_temp != after_temp:
                if after.get('temperature_filled'):
                    changes.append(f"Temp: None → {after_temp}°C (filled)")
                elif after.get('temperature_corrected'):
                    changes.append(f"Temp: {before_temp}°C → {after_temp}°C (corrected)")
            
            # Battery changes
            before_batt = before.get('Battery_Level')
            after_batt = after.get('Battery_Level')
            if before_batt != after_batt:
                if after.get('battery_filled'):
                    changes.append(f"Battery: {before_batt} → {after_batt} (filled)")
                elif after.get('battery_corrected'):
                    changes.append(f"Battery: {before_batt} → {after_batt} (corrected)")
            
            # Time changes
            before_date = before.get('Date', '')
            after_date = after.get('Date', '')
            if before_date != after_date:
                before_time = before_date.split(' ')[1] if ' ' in before_date else 'None'
                after_time = after_date.split(' ')[1] if ' ' in after_date else 'None'
                if after.get('time_filled'):
                    changes.append(f"Time: {before_time} → {after_time} (filled)")
                elif after.get('time_corrected'):
                    changes.append(f"Time: {before_time} → {after_time} (corrected)")
            
            if changes:
                correction_examples.append(f"   • {filename}: {'; '.join(changes)}")
        
        for example in correction_examples:
            print(example)
        
        if not correction_examples:
            print("   • No corrections were needed for the sample data")

    def _print_filename_format_breakdown(self):
        """Print detailed breakdown of filename formats detected"""
        print(f"\n📁 DETAILED FILENAME FORMAT ANALYSIS:")
        
        total = self.stats.get('total_processed', 0)
        
        if self.stats['filename_formats']:
            for fmt, count in self.stats['filename_formats'].most_common():
                percentage = (count / total * 100) if total > 0 else 0
                
                # Add description for each format
                descriptions = {
                    'YYMMDDAA_NNNNNN': 'Simple timelapse sequence (date + sequence number)',
                    'YYMMDDAA_ID_NNNNNN': 'Complex timelapse with identifier (date + ID + sequence)',
                    'YYMMDDAA_FrameXXXX': 'Single frame format (date + frame ID)',
                    'unknown': 'Unrecognized filename pattern'
                }
                
                desc = descriptions.get(fmt, 'Custom format')
                print(f"   • {fmt}: {count:,} ({percentage:.1f}%) - {desc}")
                
                # Show examples for each format
                if hasattr(self, 'filename_examples'):
                    examples = self.filename_examples.get(fmt, [])[:3]  # Show up to 3 examples
                    for example in examples:
                        print(f"     Example: {example}")
        else:
            print("   • No filename patterns detected")

    def _print_detailed_sample(self, results: List[Dict], phase: str):
        """Print detailed sample results for before/after comparison"""
        print(f"\n{phase} PROCESSING SAMPLE (first 10 results):")
        print("-" * 120)
        print(f"{'Filename':<25} {'Format':<20} {'Date':<20} {'Temp':<8} {'Battery':<8} {'Corrections':<25}")
        print("-" * 120)
        
        for result in results:
            filename = result.get('filename', 'unknown')[:24]
            filename_info = result.get('filename_info', {})
            fmt = filename_info.get('filename_format', 'unknown')[:19]
            date = result.get('Date', 'None')[:19]
            temp = str(result.get('Temperature_C', 'None'))[:7]
            battery = str(result.get('Battery_Level', 'None'))[:7]
            
            # Collect corrections
            corrections = []
            if result.get('temperature_filled'):
                corrections.append("temp_fill")
            if result.get('temperature_corrected'):
                corrections.append("temp_fix")
            if result.get('battery_filled'):
                corrections.append("batt_fill")
            if result.get('battery_corrected'):
                corrections.append("batt_fix")
            if result.get('time_filled'):
                corrections.append("time_fill")
            if result.get('time_corrected'):
                corrections.append("time_fix")
            
            correction_str = ",".join(corrections)[:24] if corrections else "none"
            
            print(f"{filename:<25} {fmt:<20} {date:<20} {temp:<8} {battery:<8} {correction_str:<25}")
        
        print("-" * 120)

    def _print_sample_results(self, results: List[Dict], phase: str):
        """Print sample results for debugging"""
        print(f"\n{phase} POST-PROCESSING SAMPLE:")
        for result in results:
            filename = result.get('filename', 'unknown')
            date = result.get('Date', 'None')
            temp = result.get('Temperature_C', 'None')
            battery = result.get('Battery_Level', 'None')
            
            corrections = []
            if result.get('temperature_filled'):
                corrections.append("temp_filled")
            if result.get('temperature_corrected'):
                corrections.append("temp_corrected")
            if result.get('battery_filled'):
                corrections.append("batt_filled")
            if result.get('battery_corrected'):
                corrections.append("batt_corrected")
            if result.get('time_filled'):
                corrections.append("time_filled")
            if result.get('time_corrected'):
                corrections.append("time_corrected")
            
            correction_str = f" [{', '.join(corrections)}]" if corrections else ""
            
            print(f"   {filename}: {date} | {temp}°C | {battery}{correction_str}")
    
    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process single Plotwatcher image with enhanced live time prediction"""
        path = Path(image_path)
        filename = path.name
        
        # Parse filename first
        filename_info = self.pattern_matcher.parse_filename(filename)
        
        # Base result structure
        result = {
            'ID': filename,
            'filename': filename,
            'image_path': str(path),
            'Date': None,
            'Camera_Brand': 'plotwatcher',
            'Camera_Model': None,
            'Temperature_C': None,
            'Battery_Level': None,
            'processing_method': 'plotwatcher_ocr',
            'extraction_confidence': 0.0,
            'filename_info': filename_info
        }
        
        # Update stats for filename parsing
        if filename_info.get('filename_date'):
            self.stats['filename_dates_parsed'] += 1
        if filename_info.get('filename_format') != 'unknown':
            self.stats['filename_formats'][filename_info['filename_format']] += 1
        
        if filename_info.get('sequence_identifier'):
            self.stats['sequence_identifiers'][filename_info['sequence_identifier']] += 1
        
        if not path.exists():
            return result
        
        try:
            # Extract OCR text
            ocr_data = self.ocr_processor.extract_text(str(path))
            
            if not ocr_data:
                self.stats['failed_extractions'] += 1
                return result
            
            # Extract text strings
            text_strings = [r['text'] for r in ocr_data]
            
            # Extract date
            extracted_date = self.pattern_matcher.extract_date_from_texts(text_strings)
            if not extracted_date:
                extracted_date = self.pattern_matcher.extract_date(text_strings)
            if extracted_date:
                result['Date'] = f"{extracted_date} 00:00:00"
                result['extraction_confidence'] = 0.92
                self.stats['dates_extracted'] += 1
            elif filename_info.get('filename_date'):
                result['Date'] = f"{filename_info['filename_date']} 00:00:00"
                result['extraction_confidence'] = 0.85
                result['processing_method'] = 'filename_date'
            
            # Extract time
            extracted_time = self.pattern_matcher.extract_time(text_strings)
            if extracted_time and result.get('Date'):
                date_part = result['Date'].split(' ')[0]
                result['Date'] = f"{date_part} {extracted_time}"
                result['extraction_confidence'] = 0.95
                self.stats['times_extracted'] += 1
            
            # Extract temperature and battery
            temperature_data = self.pattern_matcher.extract_temperature(text_strings)
            if temperature_data:
                result['Temperature_C'] = temperature_data['celsius']
                result['Temperature_Original'] = temperature_data['original']
                self.stats['temperatures_extracted'] += 1
                self.stats['temperature_units'][temperature_data['original_unit']] += 1
            
            battery_level = self.pattern_matcher.extract_battery_level(text_strings)
            if battery_level:
                result['Battery_Level'] = battery_level
                self.stats['battery_levels_extracted'] += 1
            
            # Check for PRO model
            full_text = " ".join(text_strings).lower()
            if 'pro' in full_text:
                result['Camera_Model'] = 'PRO'
            
            # 🆕 ENHANCED LIVE DEBUG OUTPUT WITH TIME PREDICTION
            if self.debug:
                print(f"\n🔤 PLOTWATCHER PROCESSED: {filename}")
                print(f"   📝 OCR Texts Found: {text_strings}")
                print(f"   📁 Filename Format: {filename_info.get('filename_format', 'unknown')}")
                
                if filename_info.get('filename_date'):
                    print(f"   📅 Filename Date: {filename_info['filename_date']}")
                if filename_info.get('sequence_number') is not None:
                    print(f"   🔢 Sequence Number: {filename_info['sequence_number']:06d}")
                if filename_info.get('sequence_identifier'):
                    print(f"   🆔 Sequence ID: {filename_info['sequence_identifier']}")
                
                # 🔧 FIXED: Only show prediction when we don't have OCR time
                date_key = filename_info.get('filename_date')
                seq_id = filename_info.get('sequence_identifier', 'default')
                seq_num = filename_info.get('sequence_number')
                
            # In _process_single_image debug section:
            if date_key and seq_num is not None:
                if extracted_time:
                    # We have OCR time - show it directly, no prediction needed
                    print(f"   ⏰ OCR Time Found: {extracted_time} ✅")
                    
                    # Show stable interval for reference
                    interval_sec = self.timelapse_analyzer._get_stable_interval(date_key, seq_id)
                    cache_status = "cached" if (date_key, seq_id) in self.timelapse_analyzer.cached_intervals else "calculated"
                    print(f"   ⏱️  Sequence Interval: {interval_sec}s ({interval_sec//60}m {interval_sec%60}s) [{cache_status}]")
                else:
                    # No OCR time - show prediction
                    interval_sec, predicted_time, confidence_info = self.timelapse_analyzer.get_live_prediction(
                        date_key, seq_id, seq_num
                    )
                    
                    if predicted_time:
                        cache_status = "cached" if (date_key, seq_id) in self.timelapse_analyzer.cached_intervals else "calculated"
                        print(f"   ⏰ Predicted Time: {predicted_time} (no OCR found)")
                        if interval_sec:
                            print(f"   ⏱️  Interval Used: {interval_sec}s ({interval_sec//60}m {interval_sec%60}s) [{cache_status}]")
                        if confidence_info:
                            print(f"   📊 Confidence: {confidence_info}")
                    else:
                        print(f"   ⏰ Time: No OCR found, no prediction available")
                        
                if extracted_date:
                    print(f"   📅 OCR Date: {extracted_date}")
                else:
                    print(f"   📅 OCR Date: No date found")
                
                if temperature_data:
                    print(f"   🌡️  Temperature: {temperature_data['original']} = {temperature_data['celsius']}°C")
                else:
                    print(f"   🌡️  Temperature: No temperature found")
                
                if battery_level:
                    print(f"   🔋 Battery Level: {battery_level}")
                else:
                    print(f"   🔋 Battery Level: No battery info found")
                
                print(f"   🎯 Confidence: {result['extraction_confidence']:.2f}")
            
            # Update success stats
            if result['Date']:
                self.stats['successful_extractions'] += 1
            else:
                self.stats['failed_extractions'] += 1
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            result['error_message'] = str(e)
            self.stats['failed_extractions'] += 1
        
        return result
    
    def _scan_directory(self, directory: str) -> List[str]:
        """Scan directory for image files"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        image_paths = []
        
        print(f"🔍 Scanning directory: {directory}")
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        print(f"📁 Found {len(image_paths)} images")
        return sorted(image_paths)
    
    def export_csv(self, results: List[Dict], output_path: str):
        """Export results to CSV"""
        # Flatten filename_info for CSV export
        flattened_results = []
        for result in results:
            flat_result = result.copy()
            filename_info = flat_result.pop('filename_info', {})
            
            # Add filename info as separate columns
            flat_result['filename_date'] = filename_info.get('filename_date')
            flat_result['sequence_number'] = filename_info.get('sequence_number')
            flat_result['sequence_identifier'] = filename_info.get('sequence_identifier')
            flat_result['frame_id'] = filename_info.get('frame_id')
            flat_result['has_timelapse'] = filename_info.get('has_timelapse')
            flat_result['filename_format'] = filename_info.get('filename_format')
            
            flattened_results.append(flat_result)
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
    
    def print_stats(self):
        """Enhanced statistics with detailed breakdown"""
        print("\n" + "="*70)
        print("🔍 PLOTWATCHER CAMERA TRAP OCR - PROCESSING STATISTICS")
        print("="*70)
        
        total = self.stats.get('total_processed', 0)
        successful = self.stats.get('successful_extractions', 0)
        
        print(f"📊 Total Images Processed: {total:,}")
        if total > 0:
            success_rate = (successful / total) * 100
            print(f"🎯 Success Rate: {success_rate:.1f}% ({successful:,}/{total:,})")
        
        print(f"\n📋 EXTRACTION RESULTS:")
        print(f"   ✅ Successful: {successful:,}")
        print(f"   ❌ Failed: {self.stats.get('failed_extractions', 0):,}")
        
        print(f"\n🔍 DETAILED EXTRACTION BREAKDOWN:")
        print(f"   📅 OCR Dates Extracted: {self.stats.get('dates_extracted', 0):,}")
        print(f"   ⏰ OCR Times Extracted: {self.stats.get('times_extracted', 0):,}")
        print(f"   🌡️  Temperatures Extracted: {self.stats.get('temperatures_extracted', 0):,}")
        print(f"   🔋 Battery Levels Extracted: {self.stats.get('battery_levels_extracted', 0):,}")
        print(f"   📁 Filename Dates Parsed: {self.stats.get('filename_dates_parsed', 0):,}")
        
        # Temperature unit breakdown
        if self.stats['temperature_units']:
            print(f"\n🌡️  TEMPERATURE UNITS:")
            for unit, count in self.stats['temperature_units'].most_common():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   • {unit}: {count:,} ({percentage:.1f}%)")
        
        # Filename format breakdown
        if self.stats['filename_formats']:
            print(f"\n📁 FILENAME FORMATS:")
            for fmt, count in self.stats['filename_formats'].most_common():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   • {fmt}: {count:,} ({percentage:.1f}%)")
        
        # Sequence identifier breakdown
        if self.stats['sequence_identifiers']:
            print(f"\n🆔 SEQUENCE IDENTIFIERS:")
            for seq_id, count in self.stats['sequence_identifiers'].most_common(10):
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   • {seq_id}: {count:,} ({percentage:.1f}%)")
        
        # 🔧 FIXED: Timelapse analysis results
        if hasattr(self, 'timelapse_analysis') and self.timelapse_analysis.get('detected_intervals'):
            print(f"\n⏱️  TIMELAPSE ANALYSIS:")
            detected_intervals = self.timelapse_analysis['detected_intervals']
            print(f"   🎬 Sequences Detected: {len(detected_intervals):,}")
            
            for date_key, seq_data in detected_intervals.items():
                for seq_id, interval_data in seq_data.items():
                    # Handle both old and new structure
                    if isinstance(interval_data, dict):
                        interval_sec = interval_data.get('interval_seconds', 10)
                        confidence = interval_data.get('confidence', 0.0)
                        samples = interval_data.get('total_samples', 0)
                    else:
                        # Fallback for unexpected structure
                        interval_sec = 10
                        confidence = 0.0
                        samples = 0
                    
                    print(f"   📅 {date_key} ({seq_id}):")
                    print(f"      ⏱️  Interval: {interval_sec}s ({interval_sec//60}m {interval_sec%60}s)")
                    print(f"      🎯 Confidence: {confidence:.1%} ({samples} samples)")
        
        # Post-processing statistics
        self.post_processor.print_correction_stats()
        
        if self.stats.get('processing_time'):
            rate = total / self.stats['processing_time']
            print(f"\n⚡ PERFORMANCE:")
            print(f"   🚀 Processing Speed: {rate:.1f} images/second")
            print(f"   ⏱️  Total Time: {self.stats['processing_time']:.1f} seconds")
        
        print("="*70)
    
    def print_debug_failures(self, limit: int = 10):
        """Enhanced debug failures with more detail"""
        if not self.debug or not self.debug_failures:
            return
        
        failures_to_show = self.debug_failures[:limit]
        
        print(f"\n🐛 PLOTWATCHER DEBUG FAILURES (showing {len(failures_to_show)} of {len(self.debug_failures)}):")
        for i, failure in enumerate(failures_to_show, 1):
            print(f"#{i}: {failure['filename']}")
            
            # Show what OCR found
            if failure.get('ocr_texts'):
                print(f"   📝 OCR found: {failure['ocr_texts']}")
            else:
                print(f"   📝 OCR found: No text detected")
            
            # Show failure reasons
            if failure.get('failures'):
                print(f"   ❌ Issues: {', '.join(failure['failures'])}")
            
            # Show processing steps that succeeded
            if failure.get('steps'):
                print(f"   ✅ Succeeded: {', '.join(failure['steps'])}")
            
            print()

def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plotwatcher Camera Trap OCR - Enhanced with Timelapse Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plotwatcher.py /path/to/plotwatcher/images --output results.csv
  python plotwatcher.py /path/to/images --debug
  python plotwatcher.py /path/to/images --paddleOCR --debug
        """
    )
    
    # Input
    parser.add_argument('images', help='Image files or directory')
    
    # Output
    parser.add_argument('--output', '-o', default='plotwatcher_results.csv', 
                       help='Output CSV file')
    
    # Processing
    parser.add_argument('--paddleOCR', action='store_true', 
                       help='Use PaddleOCR instead of EasyOCR')
    parser.add_argument('--confidence', type=float, default=0.3, 
                       help='OCR confidence threshold')
    
    # Debug
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )
    
    # Validate input
    if not os.path.exists(args.images):
        print(f"❌ Path not found: {args.images}")
        return 1
    
    # Select OCR engine
    ocr_engine = "paddleocr" if args.paddleOCR else "easyocr"
    
    # Validate OCR engine
    try:
        if ocr_engine == "paddleocr":
            import paddleocr
        else:
            import easyocr
    except ImportError:
        engine_name = "PaddleOCR" if ocr_engine == "paddleocr" else "EasyOCR"
        print(f"❌ {engine_name} not installed!")
        print(f"📦 Install: pip install {'paddlepaddle paddleocr' if ocr_engine == 'paddleocr' else 'easyocr'}")
        return 1
    
    print(f"🚀 Enhanced Plotwatcher Camera Trap OCR - {ocr_engine.upper()}")
    
    try:
        # Initialize extractor
        extractor = PlotwatcherExtractor(
            ocr_engine=ocr_engine,
            confidence=args.confidence,
            debug=args.debug
        )
        
        # Process images
        print(f"📸 Starting Plotwatcher image processing with {ocr_engine.upper()}...")
        start_time = time.time()
        
        results = extractor.process_images(args.images)
        
        processing_time = time.time() - start_time
        
        # Print results
        if args.debug:
            extractor.print_debug_failures()
        
        extractor.print_stats()
        
        # Export results
        extractor.export_csv(results, args.output)
        print(f"✅ Results exported to {args.output}")
        
        # Success summary
        successful = len([r for r in results if r.get('Date')])
        total = len(results)
        success_rate = (successful / total * 100) if total else 0
        
        print(f"\n🎉 Processing complete!")
        print(f"📊 {success_rate:.1f}% success rate ({successful}/{total})")
        print(f"⚡ {total/processing_time:.1f} images/second")
        
        return 0 if success_rate > 80 else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())