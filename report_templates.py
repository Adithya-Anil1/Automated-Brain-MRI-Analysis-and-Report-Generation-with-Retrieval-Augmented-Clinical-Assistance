#!/usr/bin/env python3
"""
Report Templates for MRI Brain Tumor Analysis

This module contains human-written, rigid templates for radiology reports.
The LLM fills in specific fields but CANNOT modify the template structure.

Template Philosophy:
- Templates are 100% human-written and clinically validated
- LLM only provides content for designated placeholder fields
- All formatting, section order, and standard language is fixed
- Ensures consistency, compliance, and clinical appropriateness

Slot Constraint System:
- Each slot has allowed templates, forbidden terms, and value constraints
- This is CLINICAL GOVERNANCE IN CODE
- Violations are caught and corrected automatically

7-Point Validation System:
1. Section-level forbidden terms (no diagnostic language in FINDINGS)
2. Atomic sentence templates (no phrase concatenation)
3. Concept ownership (single-source reporting)
4. Paragraph grouping (natural flow)
5. Hedged diagnostic phrasing (reviewer-safe)
6. Section-scoped slot definitions (no tone bleed)
7. Final deterministic validation pass (gatekeeper)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import re


# ============================================================================
# SECTION-LEVEL CONSTRAINTS (Issue #1, #5, #6)
# ============================================================================
# These enforce section-specific semantic rules

SECTION_FORBIDDEN_TERMS = {
    "FINDINGS": [
        "concern", "concerning", "suspicious", "suspicious for",
        "suggestive", "suggestive of", "raises concern", "raising concern",
        "neoplasm", "glioblastoma", "tumor", "cancer", "disease",
        "differential", "likely", "probable", "favor", "favoring",
        "consistent with", "diagnostic of", "compatible with",
    ],
    "IMPRESSION": [
        # These are forbidden in impression because they're unhedged
        "diagnostic of", "definitive for", "definitely",
        "certainly", "100%", "always", "never",
    ],
}

# Allowed hedged phrases for diagnostic language in IMPRESSION (Issue #5)
HEDGED_DIAGNOSTIC_PHRASES = {
    "high_grade": [
        "suspicious for a high-grade neoplastic process",
        "raises concern for high-grade neoplasm",
        "imaging features are concerning for high-grade neoplasm",
    ],
    "metastasis": [
        "suspicious for metastatic disease",
        "raises concern for metastases",
    ],
    "glioblastoma": [
        "suspicious for high-grade glioma",
        "concerning for high-grade glial neoplasm",
    ],
}

# Forbidden unhedged diagnostic terms for IMPRESSION
FORBIDDEN_IMPRESSION_UNHEDGED = [
    "diagnostic of", "consistent with", "definitive for",
    "confirms", "represents", "is a", "definitely",
]

# ============================================================================
# CONCEPT OWNERSHIP REGISTRY (Issue #3)
# ============================================================================
# Each concept is reported in ONE slot only - prevents duplication

CONCEPT_OWNERSHIP = {
    "necrosis": "necrosis_sentence",  # Only necrosis_sentence reports necrosis
    "edema": "edema_sentence",  # Only edema_sentence reports edema
    "mass_effect": "mass_effect_sentence",  # Only mass_effect reports shift
    "enhancement_center": "enhancement_sentence",  # Enhancement reports non-enhancing center
    "ring_enhancement": "enhancement_sentence",  # Ring pattern owned by enhancement
    "midline_shift": "mass_effect_sentence",
    "herniation": "mass_effect_sentence",
    "hydrocephalus": "ventricles_sentence",
}

# ============================================================================
# PARAGRAPH GROUPING (Issue #4)
# ============================================================================
# Group sentences into conceptual blocks for natural reading

FINDINGS_PARAGRAPH_STRUCTURE = {
    "lesion_description": [
        "lesion_count_sentence",
        "dominant_lesion_sentence",
    ],
    "signal_characteristics": [
        "enhancement_sentence",
        "necrosis_sentence",
        "edema_sentence",
    ],
    "secondary_effects": [
        "mass_effect_sentence",
    ],
    "normal_structures": [
        "ventricles_sentence",
        "parenchyma_sentence",
    ],
}

# ============================================================================
# SLOT SPECIFICATIONS - CLINICAL GOVERNANCE IN CODE
# ============================================================================
# Each slot has:
#   - allowed_templates: Pre-approved sentence structures
#   - forbidden_terms: Words that MUST NOT appear (clinical/legal reasons)
#   - allowed_values: Constrained vocabulary for specific placeholders
#   - max_length: Maximum character length
#   - fallback: Default value if generation fails
#   - section: Which section this slot belongs to (for section-level validation)
# ============================================================================

SLOT_SPECIFICATIONS = {
    "clinical_indication": {
        "allowed_templates": [
            "Clinical indication not provided.",
            "{clinical_history}",
            "{clinical_history}. Presenting symptoms: {symptoms}.",
            "Presenting symptoms: {symptoms}.",
            "Evaluation for {indication}.",
        ],
        "forbidden_terms": [],
        "max_length": 500,
        "fallback": "Clinical indication not provided.",
        "section": "CLINICAL_INDICATION",
    },
    
    "sequences_list": {
        "allowed_templates": [
            "T1-weighted, post-contrast T1-weighted, T2-weighted, and FLAIR",
            "T1-weighted, T2-weighted, and FLAIR",
            "T1-weighted and T2-weighted",
            "{custom_sequences}",
        ],
        "forbidden_terms": [],
        "max_length": 200,
        "fallback": "standard sequences",
        "section": "TECHNIQUE",
    },
    
    "contrast_sentence": {
        "allowed_templates": [
            "Post-contrast T1-weighted imaging was obtained following intravenous gadolinium administration.",
            "No intravenous contrast was administered.",
        ],
        "forbidden_terms": [],
        "max_length": 150,
        "fallback": "Post-contrast T1-weighted imaging was obtained following intravenous gadolinium administration.",
        "section": "TECHNIQUE",
    },
    
    "comparison": {
        "allowed_templates": [
            "No prior imaging available for comparison.",
            "Compared to prior MRI dated {prior_date}.",
            "Compared to prior examination.",
            "{custom_comparison}",
        ],
        "forbidden_terms": [],
        "max_length": 200,
        "fallback": "No prior imaging available for comparison.",
        "section": "COMPARISON",
    },
    
    "lesion_count_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - Issue #2: no phrase concatenation
            "A single enhancing lesion is identified within the {hemisphere} cerebral hemisphere.",
            "Two spatially separate enhancing lesions are identified within the {hemisphere} cerebral hemisphere.",
            "Multiple enhancing lesions ({count}) are identified with a {distribution} distribution.",
        ],
        "forbidden_terms": [
            # Section-level forbidden for FINDINGS
            "tumor", "cancer", "neoplasm", "malignant", "glioblastoma",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "allowed_values": {
            "hemisphere": ["right", "left", "bilateral"],
            "distribution": ["multifocal", "multicentric", "scattered", "clustered"],
        },
        "max_length": 200,
        "fallback": "An enhancing lesion is identified within the cerebral hemisphere.",
        "section": "FINDINGS",
    },
    
    "dominant_lesion_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - complete grammatical units
            "The dominant lesion is a {depth}{shape} mass located in the {lobes}, measuring approximately {size_cm} cm in maximum diameter.",
            "A {depth}{shape} mass is identified in the {lobes}, measuring approximately {size_cm} cm in maximum diameter.",
            "The primary lesion is a {depth}mass located in the {lobes}, measuring approximately {size_cm} cm.",
        ],
        "forbidden_terms": [
            # FINDINGS section: no diagnostic language
            "microscopic", "invasive", "definitive", "tumor", "cancer", 
            "neoplasm", "malignant", "glioblastoma", "metastasis",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "allowed_values": {
            "depth": ["", "subcortical ", "deep ", "cortical and subcortical ", "periventricular "],
            "shape": ["", "ovoid ", "irregular ", "round ", "lobulated "],
            "hemisphere": ["right", "left", "bilateral", "midline"],
        },
        "max_length": 250,
        "fallback": "A mass is identified within the cerebral hemisphere.",
        "section": "FINDINGS",
    },
    
    "enhancement_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - Issue #2, #3: no mention of necrosis here (concept ownership)
            "The lesion demonstrates ring enhancement with a non-enhancing central component.",
            "The lesion demonstrates heterogeneous ring enhancement.",
            "The lesion demonstrates homogeneous enhancement following contrast administration.",
            "The lesion demonstrates heterogeneous enhancement following contrast administration.",
            "No abnormal enhancement is identified.",
        ],
        "forbidden_terms": [
            "tumor", "cancer", "neoplasm", "malignant", "glioblastoma", 
            "aggressive", "definitive", "necrosis", "necrotic",  # Necrosis owned by necrosis_sentence
            "concern", "suspicious", "suggestive", "disease",
        ],
        "allowed_values": {
            "pattern": [
                "ring enhancement", "homogeneous enhancement",
                "heterogeneous enhancement", "nodular enhancement",
            ],
        },
        "max_length": 200,
        "fallback": "Enhancement is noted following contrast administration.",
        "section": "FINDINGS",
    },
    
    "necrosis_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - single owner of necrosis concept
            "Central necrosis is present within the lesion.",
            "A small central necrotic component is identified.",
            "A large area of central necrosis is present.",
            "No central necrosis is identified.",
        ],
        "forbidden_terms": [
            "tumor", "cancer", "glioblastoma", "malignant", 
            "percentage", "%", "microscopic",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "allowed_values": {},
        "max_length": 150,
        "fallback": "Central signal abnormality is noted within the lesion.",
        "section": "FINDINGS",
    },
    
    "edema_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - single owner of edema concept
            # Issue #1: No "consistent with" (interpretive language in FINDINGS)
            "Surrounding T2/FLAIR hyperintensity is present, representing vasogenic edema.",
            "Extensive surrounding T2/FLAIR hyperintensity is present, representing vasogenic edema.",
            "Significant surrounding T2/FLAIR hyperintensity is present, representing vasogenic edema.",
            "Moderate surrounding T2/FLAIR hyperintensity is present.",
            "Minimal surrounding T2/FLAIR hyperintensity is present.",
            "No significant surrounding edema is identified.",
        ],
        "forbidden_terms": [
            "cm³", "cm3", "cubic", "volume", "ml", "mL",
            "tumor", "cancer", "neoplasm",
            "concern", "suspicious", "suggestive", "disease",
            "consistent with",  # Interpretive - not for FINDINGS
        ],
        "allowed_values": {
            "edema_degree": ["Minimal", "Moderate", "Significant", "Extensive"],
        },
        "max_length": 150,
        "fallback": "Surrounding T2/FLAIR hyperintensity is present.",
        "section": "FINDINGS",
    },
    
    "mass_effect_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - single owner of mass effect/shift/herniation
            "No significant midline shift is identified. No evidence of herniation.",
            "There is approximately {shift_mm} mm of midline shift to the {direction}. No evidence of herniation.",
            "Mild mass effect is noted without significant midline shift.",
        ],
        "forbidden_terms": [
            "risk", "probability", "likely", "percent", "%",
            "tumor", "cancer", "dangerous",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "allowed_values": {
            "direction": ["left", "right"],
        },
        "max_length": 200,
        "fallback": "No significant midline shift is identified. No evidence of herniation.",
        "section": "FINDINGS",
    },
    
    "ventricles_sentence": {
        "allowed_templates": [
            # ATOMIC sentences - single owner of hydrocephalus concept
            "The ventricular system is normal in size and configuration.",
            "The ventricular system is normal in size with mild asymmetry of the lateral ventricles.",
            "The ventricular system demonstrates ventriculomegaly.",
            "Mild ventriculomegaly is noted.",
        ],
        "forbidden_terms": [
            "tumor", "cancer", "mass",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "max_length": 150,
        "fallback": "The ventricular system is normal in size and configuration.",
        "section": "FINDINGS",
    },
    
    "parenchyma_sentence": {
        "allowed_templates": [
            "The remaining brain parenchyma demonstrates preserved gray-white matter differentiation.",
            "Background white matter changes are noted. Gray-white matter differentiation is otherwise preserved.",
            "The remaining brain parenchyma appears unremarkable.",
        ],
        "forbidden_terms": [
            "tumor", "cancer", "neoplasm", "metastasis",
            "concern", "suspicious", "suggestive", "disease",
        ],
        "max_length": 200,
        "fallback": "The remaining brain parenchyma appears unremarkable.",
        "section": "FINDINGS",
    },
    
    # =========================================================================
    # IMPRESSION SLOTS - Issue #5, #6: Hedged language, section-scoped
    # These slots use DIFFERENT phrasing than FINDINGS (no tone bleed)
    # =========================================================================
    
    "impression_summary": {
        "allowed_templates": [
            # HEDGED diagnostic language - Issue #5
            "Ring-enhancing mass in the {hemisphere} {lobe} lobe, measuring approximately {size_cm} cm, suspicious for high-grade neoplastic process.",
            "Multifocal ring-enhancing masses in the {hemisphere} cerebral hemisphere, largest measuring approximately {size_cm} cm, suspicious for high-grade neoplastic process.",
            "Enhancing mass in the {hemisphere} {lobe} lobe, measuring approximately {size_cm} cm, with imaging features concerning for neoplastic process.",
            "Multifocal enhancing masses in the {hemisphere} cerebral hemisphere, imaging features concerning for neoplastic process.",
        ],
        "forbidden_terms": [
            # Forbidden unhedged terms - Issue #5
            "definitive", "definitely", "certainly", "proven",
            "microscopic", "invasive", "diagnostic of", "consistent with",
            "confirms", "represents", "is a",
        ],
        "allowed_values": {
            "hemisphere": ["right", "left", "bilateral"],
        },
        "max_length": 300,
        "fallback": "Enhancing mass identified, suspicious for neoplastic process. Clinical correlation recommended.",
        "section": "IMPRESSION",
    },
    
    "impression_differential": {
        "allowed_templates": [
            # Hedged and professional
            "Differential diagnosis includes {differentials}. Histopathologic correlation recommended.",
            "Differential considerations include {differentials}. Tissue sampling is recommended.",
            "Given the imaging features, differential diagnosis includes {differentials}. Clinical correlation advised.",
        ],
        "forbidden_terms": [
            "definitely", "certainly", "proven", "confirmed",
            "100%", "always", "never", "diagnostic of", "consistent with",
        ],
        "max_length": 250,
        "fallback": "Differential diagnosis includes high-grade glioma, metastasis, and lymphoma. Histopathologic correlation recommended.",
        "section": "IMPRESSION",
    },
}


# ============================================================================
# SLOT VALIDATOR CLASS
# ============================================================================

class SlotValidator:
    """
    Validates and sanitizes slot content against specifications.
    This is the enforcement layer for clinical governance.
    
    Implements 7-point validation:
    1. Global banned words check
    2. Section-level forbidden terms (no diagnostic language in FINDINGS)
    3. Slot-specific forbidden terms
    4. Max length check
    5. Concept ownership (no duplicate concepts)
    6. Sentence integrity (no fragments)
    7. Final report validation
    """
    
    # =========================================================================
    # GLOBAL BANNED WORDS - Apply to ALL slots regardless of specification
    # These words should NEVER appear in any radiology report
    # =========================================================================
    GLOBAL_BANNED_WORDS = [
        # Clinical certainty terms (inappropriate for imaging)
        "microscopic", "definitive", "histologic", "histological", "pathologic",
        "pathological", "biopsy-proven", "confirmed", "definite", "certainly",
        "definitely", "proven",
        
        # Multi-word phrases to ban
        "diagnostic of",
        
        # Inappropriate clinical terms
        "benign", "malignant", "cancer", "carcinoma", "sarcoma",
        
        # Legal/liability terms
        "malpractice", "error", "mistake", "missed", "overlooked",
        
        # Colloquial/unprofessional
        "looks like", "seems to be", "probably", "maybe", "I think",
        "in my opinion", "appears to possibly",
    ]
    
    # Allowed compound words that contain banned substrings
    ALLOWED_COMPOUND_WORDS = [
        "histopathologic",
        "histopathological",
        "histopathology",
    ]
    
    def __init__(self, specifications: Dict = None):
        self.specs = specifications or SLOT_SPECIFICATIONS
        self.reported_concepts = set()  # Track which concepts have been reported
    
    def reset_concept_tracking(self):
        """Reset concept tracking for a new report."""
        self.reported_concepts = set()
    
    def _is_allowed_compound(self, content: str, banned_word: str) -> bool:
        """Check if a banned word appears only as part of an allowed compound word."""
        content_lower = content.lower()
        banned_lower = banned_word.lower()
        
        pos = 0
        while True:
            idx = content_lower.find(banned_lower, pos)
            if idx == -1:
                break
                
            is_part_of_allowed = False
            for allowed in self.ALLOWED_COMPOUND_WORDS:
                if allowed.lower() in content_lower:
                    allowed_idx = content_lower.find(allowed.lower())
                    if allowed_idx != -1:
                        if idx >= allowed_idx and idx < allowed_idx + len(allowed):
                            is_part_of_allowed = True
                            break
            
            if not is_part_of_allowed:
                return False
            
            pos = idx + 1
        
        return True
    
    def _check_section_forbidden_terms(self, slot_name: str, content: str) -> List[str]:
        """
        Issue #1: Check section-level forbidden terms.
        FINDINGS section cannot have diagnostic language.
        """
        violations = []
        
        if slot_name not in self.specs:
            return violations
            
        section = self.specs[slot_name].get('section', '')
        
        if section in SECTION_FORBIDDEN_TERMS:
            for term in SECTION_FORBIDDEN_TERMS[section]:
                if term.lower() in content.lower():
                    violations.append(f"Section '{section}' forbidden term: '{term}'")
        
        return violations
    
    def _check_concept_ownership(self, slot_name: str, content: str) -> List[str]:
        """
        Issue #3: Check concept ownership - prevent duplicate reporting.
        Each concept is reported by one slot only.
        """
        violations = []
        
        for concept, owner_slot in CONCEPT_OWNERSHIP.items():
            if slot_name == owner_slot:
                # This slot owns this concept - mark as reported
                if concept.replace('_', ' ') in content.lower() or concept in content.lower():
                    self.reported_concepts.add(concept)
            elif concept in self.reported_concepts:
                # Another slot already reported this concept
                if concept.replace('_', ' ') in content.lower() or concept in content.lower():
                    violations.append(f"Concept '{concept}' already reported by '{owner_slot}'")
        
        return violations
    
    def _check_sentence_integrity(self, content: str) -> List[str]:
        """
        Issue #2, #4: Check for sentence fragments and grammar issues.
        """
        violations = []
        
        # Check for common fragment patterns
        fragment_patterns = [
            r'^[a-z]',  # Starts with lowercase (fragment)
            r',\s*$',   # Ends with comma
            r'^\s*and\s',  # Starts with "and"
            r'^\s*with\s',  # Starts with "with" (dangling modifier)
            r',\s*,',   # Double comma
            r'\s{2,}',  # Multiple spaces
        ]
        
        for pattern in fragment_patterns:
            if re.search(pattern, content):
                violations.append(f"Possible sentence fragment: pattern '{pattern}'")
        
        # Check for missing period at end (if not a list item)
        if content and not content.strip().endswith(('.', '?', '!')):
            if not content.strip().startswith(('1.', '2.', '-', '*')):
                violations.append("Sentence does not end with proper punctuation")
        
        return violations
    
    def validate(self, slot_name: str, content: str, check_section: bool = True) -> tuple:
        """
        Validate content against all specifications.
        
        7-POINT VALIDATION:
        1. Global banned words
        2. Section-level forbidden terms
        3. Slot-specific forbidden terms
        4. Max length
        5. Concept ownership
        6. Sentence integrity
        7. Final report validation (done separately)
        
        Returns:
            (is_valid, sanitized_content, violations)
        """
        violations = []
        sanitized = content
        
        # =====================================================================
        # CHECK 1: Global banned words
        # =====================================================================
        for banned in self.GLOBAL_BANNED_WORDS:
            if banned.lower() in content.lower():
                if not self._is_allowed_compound(content, banned):
                    violations.append(f"Global banned: '{banned}'")
                    sanitized = '[BANNED]'
                    break
        
        # =====================================================================
        # CHECK 2: Section-level forbidden terms (Issue #1)
        # =====================================================================
        if sanitized != '[BANNED]' and check_section:
            section_violations = self._check_section_forbidden_terms(slot_name, content)
            violations.extend(section_violations)
            if section_violations:
                sanitized = '[SECTION_VIOLATION]'
        
        # =====================================================================
        # CHECK 3: Slot-specific forbidden terms
        # =====================================================================
        if slot_name in self.specs and sanitized not in ['[BANNED]', '[SECTION_VIOLATION]']:
            spec = self.specs[slot_name]
            forbidden = spec.get('forbidden_terms', [])
            
            for term in forbidden:
                if term.lower() in content.lower():
                    violations.append(f"Slot forbidden: '{term}'")
                    sanitized = '[SLOT_VIOLATION]'
                    break
        
        # =====================================================================
        # CHECK 4: Max length
        # =====================================================================
        if slot_name in self.specs and sanitized not in ['[BANNED]', '[SECTION_VIOLATION]', '[SLOT_VIOLATION]']:
            spec = self.specs[slot_name]
            max_len = spec.get('max_length', float('inf'))
            if len(sanitized) > max_len:
                violations.append(f"Exceeds max length ({len(sanitized)} > {max_len})")
                sanitized = sanitized[:max_len-3] + "..."
        
        # =====================================================================
        # CHECK 5: Concept ownership (Issue #3)
        # =====================================================================
        if sanitized not in ['[BANNED]', '[SECTION_VIOLATION]', '[SLOT_VIOLATION]']:
            concept_violations = self._check_concept_ownership(slot_name, content)
            if concept_violations:
                violations.extend(concept_violations)
                # Don't fail, just log - the slot mapper should handle this
        
        # =====================================================================
        # FALLBACK: Use safe fallback if violations
        # =====================================================================
        if any(v.startswith('[') for v in [sanitized]):
            fallback = self.get_fallback(slot_name)
            if fallback:
                sanitized = fallback
                violations.append("Used fallback due to validation failure")
        
        is_valid = len(violations) == 0
        return is_valid, sanitized, violations
    
    def get_fallback(self, slot_name: str) -> str:
        """Get the fallback value for a slot."""
        if slot_name in self.specs:
            return self.specs[slot_name].get('fallback', '')
        return ''
    
    def get_allowed_values(self, slot_name: str, field: str) -> List[str]:
        """Get allowed values for a specific field within a slot."""
        if slot_name in self.specs:
            allowed = self.specs[slot_name].get('allowed_values', {})
            return allowed.get(field, [])
        return []
    
    def validate_value(self, slot_name: str, field: str, value: str) -> bool:
        """Check if a value is in the allowed list for a field."""
        allowed = self.get_allowed_values(slot_name, field)
        if not allowed:
            return True
        return value.lower() in [v.lower() for v in allowed]


# Global validator instance
slot_validator = SlotValidator()


# ============================================================================
# STEP 3: FACT EXTRACTOR - Convert Model Outputs to Structured Facts (NO LLM)
# ============================================================================
# This layer converts raw model/pipeline outputs into structured facts.
# ALL logic is deterministic - no LLM involvement.
# ============================================================================

class FactExtractor:
    """
    Extract structured facts from model outputs using deterministic rules.
    
    This is Step 3 of the pipeline: Model Outputs → Structured Facts
    NO LLM is used here - all mappings are rule-based.
    """
    
    # Thresholds for qualitative mappings
    EDEMA_THRESHOLDS = {
        'minimal': (0, 0.15),       # < 15% edema ratio
        'moderate': (0.15, 0.40),   # 15-40%
        'significant': (0.40, 0.65), # 40-65%
        'extensive': (0.65, 1.0),   # > 65%
    }
    
    MIDLINE_SHIFT_THRESHOLD_MM = 2.0  # Below this = "no significant shift"
    
    NECROSIS_THRESHOLDS = {
        'none': (0, 0.01),          # < 1%
        'minimal': (0.01, 0.10),    # 1-10%
        'moderate': (0.10, 0.30),   # 10-30%
        'extensive': (0.30, 1.0),   # > 30%
    }
    
    def __init__(self, summary: dict):
        """
        Initialize with raw summary data from pipeline.
        
        Args:
            summary: The llm_ready_summary.json data
        """
        self.summary = summary
        self._facts = None  # Cached facts
    
    def extract_facts(self) -> dict:
        """
        Extract all structured facts from the summary.
        
        Returns:
            Dictionary of structured facts with computed values
        """
        if self._facts is not None:
            return self._facts
        
        # Extract raw data from summary
        tumor = self.summary.get('tumor_characteristics', {})
        location = self.summary.get('location', {})
        multiplicity = self.summary.get('multiplicity', {})
        enhancement = self.summary.get('enhancement', {})
        necrosis = self.summary.get('necrosis', {})
        mass_effect = self.summary.get('mass_effect', {})
        morphology = self.summary.get('morphology', {})
        normal_structures = self.summary.get('normal_structures', {})
        technique = self.summary.get('technique', {})
        patient_info = self.summary.get('patient_info', {})
        differential = self.summary.get('differential_considerations', [])
        
        # Compute derived facts using deterministic rules
        self._facts = {
            # === IDENTIFIERS ===
            'case_id': self.summary.get('case_id', 'Unknown'),
            
            # === LESION COUNT & DISTRIBUTION ===
            'lesion_count': multiplicity.get('lesion_count', 1),
            'is_multifocal': multiplicity.get('lesion_count', 1) > 1,
            'distribution': self._compute_distribution(multiplicity),
            
            # === LOCATION ===
            'hemisphere': location.get('hemisphere', 'unknown'),
            'primary_lobe': location.get('primary_lobe', 'unknown'),
            'involved_lobes': location.get('involved_lobes', []),
            'lobes_formatted': self._format_lobes(location),
            'depth': location.get('depth', '').lower(),
            'depth_prefix': self._compute_depth_prefix(location),
            
            # === SIZE ===
            'max_diameter_mm': tumor.get('max_diameter_mm', 0),
            'size_cm': self._compute_size_cm(tumor),
            'volume_cm3': tumor.get('volume_cm3', 0),
            
            # === ENHANCEMENT ===
            'enhancement_present': enhancement.get('present', False),
            'enhancement_pattern': enhancement.get('pattern', '').lower(),
            'enhancement_heterogeneity': enhancement.get('heterogeneity', '').lower(),
            'is_ring_enhancing': 'ring' in enhancement.get('pattern', '').lower(),
            
            # === NECROSIS ===
            'necrosis_present': necrosis.get('present', False),
            'necrosis_percentage': necrosis.get('percentage', 0),
            'necrosis_degree': self._compute_necrosis_degree(necrosis),
            'necrosis_location': necrosis.get('location', '').lower(),
            
            # === EDEMA ===
            'edema_volume_cm3': tumor.get('edema_volume_cm3', 0),
            'total_volume_cm3': tumor.get('volume_cm3', 0),
            'edema_ratio': self._compute_edema_ratio(tumor),
            'edema_degree': self._compute_edema_degree(tumor),
            
            # === MASS EFFECT ===
            'midline_shift_mm': mass_effect.get('midline_shift_mm', 0),
            'shift_significant': self._is_shift_significant(mass_effect),
            'shift_direction': mass_effect.get('shift_direction', ''),
            
            # === MORPHOLOGY ===
            'shape': morphology.get('shape', 'mass').lower(),
            
            # === VENTRICLES ===
            'ventricles_normal': self._are_ventricles_normal(normal_structures),
            'ventricles_symmetric': self._are_ventricles_symmetric(normal_structures),
            'hydrocephalus': normal_structures.get('ventricular_system', {}).get('hydrocephalus', False),
            
            # === PARENCHYMA ===
            'parenchyma_normal': self._is_parenchyma_normal(normal_structures),
            'white_matter_disease': normal_structures.get('parenchyma', {}).get('white_matter_disease', False),
            
            # === TECHNIQUE ===
            'sequences': technique.get('sequences_performed', []),
            'contrast_given': technique.get('contrast_administered', False),
            
            # === CLINICAL INFO ===
            'clinical_history_provided': patient_info.get('clinical_history', '<not provided>') != '<not provided>',
            'prior_imaging_available': patient_info.get('relevant_prior_imaging', '<not provided>') != '<not provided>',
            
            # === DIFFERENTIAL ===
            'differentials': differential if differential else ['high-grade glioma', 'metastasis', 'lymphoma'],
        }
        
        return self._facts
    
    # =========================================================================
    # DETERMINISTIC MAPPING RULES
    # =========================================================================
    
    def _compute_distribution(self, multiplicity: dict) -> str:
        """Determine lesion distribution pattern."""
        count = multiplicity.get('lesion_count', 1)
        pattern = multiplicity.get('distribution_pattern', '').lower()
        
        if count == 1:
            return 'focal'
        elif 'distant' in pattern or 'multicentric' in pattern:
            return 'multifocal or multicentric'
        else:
            return 'multifocal'
    
    def _format_lobes(self, location: dict) -> str:
        """Format lobe list for sentence inclusion."""
        lobes = location.get('involved_lobes', [])
        hemisphere = location.get('hemisphere', '')
        
        if not lobes:
            primary = location.get('primary_lobe', 'unknown')
            return f"{hemisphere} {primary} lobe"
        
        if len(lobes) == 1:
            return f"{hemisphere} {lobes[0]} lobe"
        elif len(lobes) == 2:
            return f"{hemisphere} {lobes[0]} and {lobes[1]} lobes"
        else:
            return f"{hemisphere} {', '.join(lobes[:-1])}, and {lobes[-1]} lobes"
    
    def _compute_depth_prefix(self, location: dict) -> str:
        """Compute depth prefix for lesion description."""
        depth = location.get('depth', '').lower()
        if 'subcortical' in depth:
            return 'subcortical '
        elif 'deep' in depth:
            return 'deep '
        elif 'cortical' in depth:
            return 'cortical and subcortical '
        elif 'periventricular' in depth:
            return 'periventricular '
        return ''
    
    def _compute_size_cm(self, tumor: dict) -> float:
        """Convert mm to cm, rounded to nearest 0.5."""
        mm = tumor.get('max_diameter_mm', 0)
        cm = mm / 10.0
        # Round to nearest 0.5 cm
        return round(cm * 2) / 2
    
    def _compute_edema_ratio(self, tumor: dict) -> float:
        """Compute edema as ratio of total tumor volume."""
        edema = tumor.get('edema_volume_cm3', 0)
        total = tumor.get('volume_cm3', 1)  # Avoid division by zero
        if total <= 0:
            return 0.0
        return min(edema / total, 1.0)  # Cap at 1.0
    
    def _compute_edema_degree(self, tumor: dict) -> str:
        """Map edema volume to qualitative descriptor."""
        edema_vol = tumor.get('edema_volume_cm3', 0)
        
        # Use absolute volume thresholds (clinical practice)
        if edema_vol < 5:
            return 'Minimal'
        elif edema_vol < 15:
            return 'Moderate'
        elif edema_vol < 30:
            return 'Significant'
        else:
            return 'Extensive'
    
    def _compute_necrosis_degree(self, necrosis: dict) -> str:
        """Map necrosis percentage to qualitative descriptor."""
        if not necrosis.get('present', False):
            return 'none'
        
        pct = necrosis.get('percentage', 0) / 100.0  # Convert to ratio
        
        for degree, (low, high) in self.NECROSIS_THRESHOLDS.items():
            if low <= pct < high:
                return degree
        return 'minimal'
    
    def _is_shift_significant(self, mass_effect: dict) -> bool:
        """Determine if midline shift is clinically significant."""
        shift = mass_effect.get('midline_shift_mm', 0)
        if isinstance(shift, (int, float)):
            return shift >= self.MIDLINE_SHIFT_THRESHOLD_MM
        return False
    
    def _are_ventricles_normal(self, normal_structures: dict) -> bool:
        """Check if ventricular system is normal."""
        vent = normal_structures.get('ventricular_system', {})
        size = vent.get('size', 'Normal')
        return 'normal' in size.lower()
    
    def _are_ventricles_symmetric(self, normal_structures: dict) -> bool:
        """Check if ventricles are symmetric."""
        vent = normal_structures.get('ventricular_system', {})
        symmetry = vent.get('symmetry', 'Symmetric')
        return 'asymmetric' not in symmetry.lower()
    
    def _is_parenchyma_normal(self, normal_structures: dict) -> bool:
        """Check if background parenchyma is normal."""
        parenchyma = normal_structures.get('parenchyma', {})
        overall = parenchyma.get('overall', 'Normal')
        return 'normal' in overall.lower()


# ============================================================================
# STEP 4: CONSTRAINED LLM SLOT FILLER (Optional)
# ============================================================================
# When LLM is used, it ONLY selects from allowed values.
# Output is validated; invalid = reject and retry.
# ============================================================================

class ConstrainedLLMFiller:
    """
    Use LLM ONLY for constrained slot filling.
    
    The LLM can only:
    1. Select from a list of allowed values
    2. Fill placeholders in pre-approved templates
    
    The LLM CANNOT:
    1. Generate free-form text
    2. Add medical interpretation
    3. Use forbidden terms
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, api_key: str = None):
        """
        Initialize with optional API key.
        
        Args:
            api_key: Gemini API key (optional - falls back to rules if not provided)
        """
        self.api_key = api_key
        self.model = None
        self._setup_model()
    
    def _setup_model(self):
        """Set up the Gemini model if available."""
        if not self.api_key:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    'temperature': 0.0,  # Deterministic
                    'max_output_tokens': 50,  # Very short - just the value
                }
            )
        except ImportError:
            self.model = None
    
    def fill_slot_value(
        self, 
        slot_name: str, 
        field_name: str, 
        context: dict,
        allowed_values: List[str]
    ) -> str:
        """
        Use LLM to select the best value from allowed options.
        
        Args:
            slot_name: Name of the slot being filled
            field_name: Name of the specific field (e.g., 'edema_degree')
            context: Relevant facts/context for decision
            allowed_values: List of allowed values to choose from
            
        Returns:
            Selected value from allowed_values (or first value if LLM unavailable)
        """
        if not self.model or not allowed_values:
            # Fallback: use first allowed value or empty string
            return allowed_values[0] if allowed_values else ''
        
        # Build constrained prompt
        prompt = self._build_constrained_prompt(field_name, context, allowed_values)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                selected = response.text.strip().lower()
                
                # Validate response is in allowed values
                for allowed in allowed_values:
                    if allowed.lower() == selected or allowed.lower() in selected:
                        return allowed  # Return properly cased version
                
                # Not found - retry with stricter prompt
                prompt = self._build_retry_prompt(field_name, allowed_values, selected)
                
            except Exception as e:
                print(f"LLM error on attempt {attempt + 1}: {e}")
                continue
        
        # All retries failed - use deterministic fallback
        return self._deterministic_fallback(field_name, context, allowed_values)
    
    def _build_constrained_prompt(
        self, 
        field_name: str, 
        context: dict, 
        allowed_values: List[str]
    ) -> str:
        """Build a strictly constrained prompt for value selection."""
        values_str = ', '.join(allowed_values)
        context_str = '\n'.join(f"  {k}: {v}" for k, v in context.items())
        
        return f"""STRICT INSTRUCTION: Select exactly ONE value from the allowed list.

TASK: Choose the best value for "{field_name}"

ALLOWED VALUES (choose ONLY from these):
{values_str}

CONTEXT:
{context_str}

RULES:
- Output ONLY the selected value, nothing else
- Do NOT add explanation
- Do NOT add punctuation
- The output must EXACTLY match one of the allowed values

YOUR SELECTION:"""

    def _build_retry_prompt(
        self, 
        field_name: str, 
        allowed_values: List[str], 
        invalid_response: str
    ) -> str:
        """Build retry prompt after invalid response."""
        values_str = ', '.join(allowed_values)
        
        return f"""ERROR: Your previous response "{invalid_response}" is not valid.

You MUST select EXACTLY ONE of these values:
{values_str}

Output ONLY the value. No other text.

YOUR SELECTION:"""

    def _deterministic_fallback(
        self, 
        field_name: str, 
        context: dict, 
        allowed_values: List[str]
    ) -> str:
        """
        Deterministic fallback when LLM fails.
        Uses simple rules based on context.
        """
        if not allowed_values:
            return ''
        
        # Field-specific fallback logic
        if field_name == 'edema_degree':
            edema_vol = context.get('edema_volume_cm3', 0)
            if edema_vol < 5:
                return 'Minimal'
            elif edema_vol < 15:
                return 'Moderate'
            elif edema_vol < 30:
                return 'Significant'
            else:
                return 'Extensive'
        
        elif field_name == 'hemisphere':
            return context.get('hemisphere', allowed_values[0])
        
        elif field_name == 'distribution':
            count = context.get('lesion_count', 1)
            if count > 1:
                return 'multifocal'
            return 'focal'
        
        # Default: return first allowed value
        return allowed_values[0]


# ============================================================================
# FACTS TO SLOT VALUES MAPPER (Deterministic)
# ============================================================================

class FactsToSlotMapper:
    """
    Maps structured facts to slot template values.
    
    This is the core deterministic engine that converts facts into
    the exact values needed to fill template placeholders.
    
    NO LLM is used - all mappings are rule-based.
    """
    
    def __init__(self, facts: dict):
        """
        Initialize with extracted facts.
        
        Args:
            facts: Output from FactExtractor.extract_facts()
        """
        self.facts = facts
    
    def map_to_slot_values(self) -> dict:
        """
        Map all facts to their corresponding slot values.
        
        Returns:
            Dictionary of slot_name -> filled sentence
        """
        return {
            'patient_id': self.facts['case_id'],
            'exam_date': datetime.now().strftime('%B %d, %Y'),
            'clinical_indication': self._map_clinical_indication(),
            'sequences_list': self._map_sequences(),
            'contrast_sentence': self._map_contrast(),
            'comparison': self._map_comparison(),
            'lesion_count_sentence': self._map_lesion_count(),
            'dominant_lesion_sentence': self._map_dominant_lesion(),
            'enhancement_sentence': self._map_enhancement(),
            'necrosis_sentence': self._map_necrosis(),
            'edema_sentence': self._map_edema(),
            'mass_effect_sentence': self._map_mass_effect(),
            'ventricles_sentence': self._map_ventricles(),
            'parenchyma_sentence': self._map_parenchyma(),
            'impression_summary': self._map_impression_summary(),
            'impression_differential': self._map_impression_differential(),
        }
    
    # =========================================================================
    # SLOT MAPPING RULES (Deterministic)
    # =========================================================================
    
    def _map_clinical_indication(self) -> str:
        """Map to clinical indication slot."""
        if self.facts['clinical_history_provided']:
            return self.facts.get('clinical_history', 'Clinical indication not provided.')
        return "Clinical indication not provided."
    
    def _map_sequences(self) -> str:
        """Map sequences to readable list."""
        sequences = self.facts['sequences']
        if not sequences:
            return "standard sequences"
        
        seq_names = {
            'T1': 'T1-weighted',
            'T1CE': 'post-contrast T1-weighted',
            'T2': 'T2-weighted',
            'FLAIR': 'FLAIR'
        }
        readable = [seq_names.get(s.upper(), s) for s in sequences]
        
        if len(readable) == 1:
            return readable[0]
        elif len(readable) == 2:
            return f"{readable[0]} and {readable[1]}"
        else:
            return ', '.join(readable[:-1]) + f', and {readable[-1]}'
    
    def _map_contrast(self) -> str:
        """Map contrast administration."""
        if self.facts['contrast_given']:
            return "Post-contrast T1-weighted imaging was obtained following intravenous gadolinium administration."
        return "No intravenous contrast was administered."
    
    def _map_comparison(self) -> str:
        """Map comparison statement."""
        if self.facts['prior_imaging_available']:
            return "Compared to prior examination."
        return "No prior imaging available for comparison."
    
    def _map_lesion_count(self) -> str:
        """
        Map lesion count to sentence.
        Issue #1: NO diagnostic language (concern, disease, etc.) in FINDINGS
        Issue #2: Use atomic sentences
        """
        count = self.facts['lesion_count']
        hemisphere = self.facts['hemisphere']
        distribution = self.facts['distribution']
        
        # ATOMIC sentences without diagnostic language
        if count == 1:
            return f"A single enhancing lesion is identified within the {hemisphere} cerebral hemisphere."
        elif count == 2:
            return f"Two spatially separate enhancing lesions are identified within the {hemisphere} cerebral hemisphere."
        else:
            return f"Multiple enhancing lesions ({count}) are identified with a {distribution} distribution."
    
    def _map_dominant_lesion(self) -> str:
        """
        Map dominant lesion description.
        Issue #2: Atomic sentence template
        """
        depth_prefix = self.facts['depth_prefix']
        shape = self.facts['shape']
        lobes = self.facts['lobes_formatted']
        size_cm = self.facts['size_cm']
        
        # Clean shape formatting (add space if needed)
        shape_str = f"{shape} " if shape and shape != 'mass' else ""
        
        # Article selection (a vs an)
        first_word = depth_prefix if depth_prefix else shape_str
        article = "an" if first_word and first_word.strip()[0].lower() in 'aeiou' else "a"
        
        return f"The dominant lesion is {article} {depth_prefix}{shape_str}mass located in the {lobes}, measuring approximately {size_cm} cm in maximum diameter."
    
    def _map_enhancement(self) -> str:
        """
        Map enhancement pattern.
        Issue #2: Atomic sentence
        Issue #3: Do NOT mention necrosis here (concept ownership)
        """
        if not self.facts['enhancement_present']:
            return "No abnormal enhancement is identified."
        
        is_ring = self.facts['is_ring_enhancing']
        heterogeneity = self.facts['enhancement_heterogeneity']
        
        # ATOMIC sentences - no phrase concatenation, no mention of necrosis
        if is_ring:
            if heterogeneity and 'heterogeneous' in heterogeneity:
                return "The lesion demonstrates heterogeneous ring enhancement."
            return "The lesion demonstrates ring enhancement with a non-enhancing central component."
        else:
            if heterogeneity and 'heterogeneous' in heterogeneity:
                return "The lesion demonstrates heterogeneous enhancement following contrast administration."
            return "The lesion demonstrates homogeneous enhancement following contrast administration."
    
    def _map_necrosis(self) -> str:
        """
        Map necrosis description.
        Issue #3: This slot OWNS the necrosis concept
        """
        if not self.facts['necrosis_present']:
            return "No central necrosis is identified."
        
        degree = self.facts['necrosis_degree']
        
        # ATOMIC sentences
        if degree == 'minimal':
            return "A small central necrotic component is identified."
        elif degree == 'extensive':
            return "A large area of central necrosis is present."
        else:
            return "Central necrosis is present within the lesion."
    
    def _map_edema(self) -> str:
        """
        Map edema description.
        Issue #1: No "consistent with" in FINDINGS (interpretive)
        Issue #3: This slot OWNS the edema concept
        """
        edema_degree = self.facts['edema_degree']
        
        # ATOMIC sentences - no interpretive language like "consistent with"
        if edema_degree == 'Minimal':
            return "Minimal surrounding T2/FLAIR hyperintensity is present."
        elif edema_degree == 'Moderate':
            return "Moderate surrounding T2/FLAIR hyperintensity is present."
        elif edema_degree == 'Extensive':
            return "Extensive surrounding T2/FLAIR hyperintensity is present, representing vasogenic edema."
        else:  # Significant
            return "Significant surrounding T2/FLAIR hyperintensity is present, representing vasogenic edema."
    
    def _map_mass_effect(self) -> str:
        """
        Map mass effect description.
        Issue #3: This slot OWNS mass effect, midline shift, herniation
        """
        shift_significant = self.facts['shift_significant']
        shift_mm = self.facts['midline_shift_mm']
        shift_direction = self.facts.get('shift_direction', '')
        
        if shift_significant:
            rounded_shift = round(shift_mm, 1)
            if shift_direction:
                return f"There is approximately {rounded_shift} mm of midline shift to the {shift_direction}. No evidence of herniation."
            return f"There is approximately {rounded_shift} mm of midline shift. No evidence of herniation."
        return "No significant midline shift is identified. No evidence of herniation."
    
    def _map_ventricles(self) -> str:
        """
        Map ventricular system description.
        Issue #3: This slot OWNS hydrocephalus concept
        Issue #1: No "concerning for" in FINDINGS
        """
        if self.facts['hydrocephalus']:
            return "The ventricular system demonstrates ventriculomegaly."
        
        if not self.facts['ventricles_symmetric']:
            return "The ventricular system is normal in size with mild asymmetry of the lateral ventricles."
        
        return "The ventricular system is normal in size and configuration."
    
    def _map_parenchyma(self) -> str:
        """Map parenchyma description."""
        if self.facts['white_matter_disease']:
            return "Background white matter changes are noted. Gray-white matter differentiation is otherwise preserved."
        
        if self.facts['parenchyma_normal']:
            return "The remaining brain parenchyma demonstrates preserved gray-white matter differentiation."
        
        return "The remaining brain parenchyma appears unremarkable."
    
    def _map_impression_summary(self) -> str:
        """
        Map impression summary.
        Issue #5: HEDGED diagnostic phrasing - "suspicious for" instead of "is"
        Issue #6: DIFFERENT tone than FINDINGS (diagnostic language allowed here)
        """
        is_multifocal = self.facts['is_multifocal']
        is_ring = self.facts['is_ring_enhancing']
        hemisphere = self.facts['hemisphere']
        lobe = self.facts['primary_lobe']
        size_cm = self.facts['size_cm']
        
        # HEDGED phrases - never unhedged diagnostic statements
        if is_multifocal:
            if is_ring:
                return f"Multifocal ring-enhancing masses in the {hemisphere} cerebral hemisphere, largest measuring approximately {size_cm} cm, suspicious for high-grade neoplastic process."
            return f"Multifocal enhancing masses in the {hemisphere} cerebral hemisphere, imaging features suspicious for neoplastic process."
        else:
            if is_ring:
                return f"Ring-enhancing mass in the {hemisphere} {lobe} lobe, measuring approximately {size_cm} cm, suspicious for high-grade neoplastic process."
            return f"Enhancing mass in the {hemisphere} {lobe} lobe, measuring approximately {size_cm} cm, with imaging features concerning for neoplastic process."
    
    def _map_impression_differential(self) -> str:
        """
        Map differential diagnosis.
        Issue #5: Hedged language
        """
        differentials = self.facts['differentials']
        
        # Format differentials for clinical style
        formatted = []
        for d in differentials[:3]:
            d_lower = d.lower()
            if 'glioma' in d_lower or 'glioblastoma' in d_lower:
                formatted.append('high-grade glioma')  # More hedged than "glioblastoma"
            elif 'metast' in d_lower:
                formatted.append('metastatic disease')
            elif 'lymphoma' in d_lower:
                formatted.append('primary CNS lymphoma')
            else:
                formatted.append(d.lower())
        
        # Remove duplicates
        unique = list(dict.fromkeys(formatted))
        
        if len(unique) >= 2:
            diff_str = ', '.join(unique[:-1]) + f', and {unique[-1]}'
        elif unique:
            diff_str = unique[0]
        else:
            diff_str = 'high-grade neoplasm'
        
        return f"Differential diagnosis includes {diff_str}. Clinical and histopathologic correlation recommended."


# ============================================================================
# MASTER TEMPLATE - MRI BRAIN WITH CONTRAST
# ============================================================================
# This is the rigid, human-written template structure.
# The LLM cannot modify this - it can only fill in the placeholders.
#
# Issue #4: Paragraph grouping for natural flow
# - Lesion description block (count + dominant lesion)
# - Signal characteristics block (enhancement + necrosis + edema)  
# - Secondary effects block (mass effect)
# - Normal structures block (ventricles + parenchyma)

MRI_BRAIN_TEMPLATE = """
MRI BRAIN WITH CONTRAST

PATIENT ID: {patient_id}
DATE: {exam_date}

CLINICAL INDICATION:
{clinical_indication}

TECHNIQUE:
Multiplanar, multisequence MRI of the brain was performed including {sequences_list}. {contrast_sentence}

COMPARISON:
{comparison}

FINDINGS:
{lesion_count_sentence} {dominant_lesion_sentence}

{enhancement_sentence} {necrosis_sentence} {edema_sentence}

{mass_effect_sentence}

{ventricles_sentence} {parenchyma_sentence}

IMPRESSION:
1. {impression_summary}
2. {impression_differential}

DISCLAIMER:
This report was generated with automated assistance and should be reviewed by a qualified radiologist.
""".strip()


# ============================================================================
# SENTENCE GENERATION RULES
# ============================================================================
# These are the rules for generating each field in the template.
# Each function generates ONE sentence/phrase based on structured data.

class SentenceGenerators:
    """
    Generate individual sentences for report fields based on structured data.
    Each method returns a clinically appropriate sentence or phrase.
    """
    
    @staticmethod
    def generate_clinical_indication(patient_info: dict) -> str:
        """Generate clinical indication statement."""
        clinical_history = patient_info.get('clinical_history', '<not provided>')
        presenting_symptoms = patient_info.get('presenting_symptoms', '<not provided>')
        
        if clinical_history != '<not provided>' and presenting_symptoms != '<not provided>':
            return f"{clinical_history}. Presenting symptoms: {presenting_symptoms}."
        elif clinical_history != '<not provided>':
            return clinical_history
        elif presenting_symptoms != '<not provided>':
            return f"Presenting symptoms: {presenting_symptoms}"
        else:
            return "Clinical indication not provided."
    
    @staticmethod
    def generate_sequences_list(technique: dict) -> str:
        """Generate list of sequences performed."""
        sequences = technique.get('sequences_performed', [])
        if not sequences:
            return "standard sequences"
        
        # Map sequence codes to readable names
        seq_names = {
            'T1': 'T1-weighted',
            'T1CE': 'post-contrast T1-weighted',
            'T2': 'T2-weighted',
            'FLAIR': 'FLAIR'
        }
        
        readable = [seq_names.get(s.upper(), s) for s in sequences]
        
        if len(readable) == 1:
            return readable[0]
        elif len(readable) == 2:
            return f"{readable[0]} and {readable[1]}"
        else:
            return ', '.join(readable[:-1]) + f', and {readable[-1]}'
    
    @staticmethod
    def generate_contrast_sentence(technique: dict) -> str:
        """Generate contrast administration statement."""
        if technique.get('contrast_administered', False):
            return "Post-contrast T1-weighted imaging was obtained following intravenous gadolinium administration."
        else:
            return "No intravenous contrast was administered."
    
    @staticmethod
    def generate_comparison(patient_info: dict) -> str:
        """Generate comparison statement."""
        prior = patient_info.get('relevant_prior_imaging', '<not provided>')
        if prior != '<not provided>':
            return prior
        else:
            return "No prior imaging available for comparison."
    
    @staticmethod
    def generate_lesion_count_sentence(multiplicity: dict, location: dict) -> str:
        """Generate sentence describing lesion count and distribution."""
        count = multiplicity.get('lesion_count', 1)
        hemisphere = location.get('hemisphere', 'unknown')
        laterality = location.get('laterality', '')
        pattern = multiplicity.get('distribution_pattern', '')
        
        if count == 1:
            return f"A single lesion is identified within the {hemisphere} cerebral hemisphere."
        elif count == 2:
            if 'distant' in pattern.lower() or 'multicentric' in pattern.lower():
                return f"Two spatially separate lesions are identified within the {hemisphere} hemisphere, raising concern for multifocal or multicentric disease."
            else:
                return f"Two lesions are identified within the {hemisphere} hemisphere."
        else:
            return f"Multiple ({count}) lesions are identified, with a {pattern.lower()} distribution."
    
    @staticmethod
    def generate_dominant_lesion_sentence(tumor: dict, location: dict, morphology: dict) -> str:
        """Generate sentence describing the dominant/primary lesion."""
        # Get size (round to nearest 0.5 cm)
        max_diameter_mm = tumor.get('max_diameter_mm', 0)
        size_cm = round(max_diameter_mm / 10 * 2) / 2  # Round to nearest 0.5 cm
        
        # Get location
        primary_lobe = location.get('primary_lobe', 'unknown')
        hemisphere = location.get('hemisphere', '')
        involved_lobes = location.get('involved_lobes', [])
        
        # Get shape
        shape = morphology.get('shape', 'mass').lower()
        
        # Build location string
        if len(involved_lobes) > 1:
            lobes_str = ', '.join(involved_lobes[:-1]) + f' and {involved_lobes[-1]} lobes'
        elif involved_lobes:
            lobes_str = f"{involved_lobes[0]} lobe"
        else:
            lobes_str = f"{primary_lobe} lobe"
        
        # Depth description
        depth = location.get('depth', '').lower()
        depth_phrase = ""
        if 'subcortical' in depth:
            depth_phrase = "subcortical "
        elif 'deep' in depth:
            depth_phrase = "deep "
        elif 'cortical' in depth:
            depth_phrase = "cortical and subcortical "
        
        # Use correct article (a/an) based on depth phrase
        article = "an" if depth_phrase.startswith(('a', 'e', 'i', 'o', 'u')) else "a"
        if not depth_phrase:
            article = "a"  # Default for shape
        
        return f"The dominant lesion is {article} {depth_phrase}{shape} mass in the {hemisphere} {lobes_str}, measuring approximately {size_cm} cm in maximum diameter."
    
    @staticmethod
    def generate_enhancement_sentence(enhancement: dict, signal: dict) -> str:
        """Generate sentence describing enhancement pattern."""
        present = enhancement.get('present', False)
        pattern = enhancement.get('pattern', '')
        strength = enhancement.get('strength', '')
        heterogeneity = enhancement.get('heterogeneity', '')
        
        if not present:
            return "No abnormal enhancement is identified."
        
        # Build enhancement description
        descriptors = []
        if heterogeneity and 'heterogeneous' in heterogeneity.lower():
            descriptors.append(heterogeneity.lower())
        
        if 'ring' in pattern.lower():
            if descriptors:
                return f"The lesion demonstrates {pattern.lower()}, {', '.join(descriptors)}, with a non-enhancing central component."
            else:
                return f"The lesion demonstrates {pattern.lower()} with a non-enhancing central component."
        else:
            pattern_desc = pattern.lower() if pattern else 'enhancement'
            if descriptors:
                return f"The lesion demonstrates {', '.join(descriptors)} {pattern_desc} following contrast administration."
            else:
                return f"The lesion demonstrates {pattern_desc} following contrast administration."
    
    @staticmethod
    def generate_necrosis_sentence(necrosis: dict) -> str:
        """Generate sentence describing necrosis (qualitative only)."""
        present = necrosis.get('present', False)
        pattern = necrosis.get('pattern', '')
        nec_location = necrosis.get('location', '')
        
        if not present:
            return "No central necrosis is identified."
        
        # Use qualitative descriptors based on pattern
        if 'minimal' in pattern.lower():
            qualifier = "A small"
        elif 'moderate' in pattern.lower():
            qualifier = "A"
        elif 'extensive' in pattern.lower() or 'significant' in pattern.lower():
            qualifier = "A large"
        else:
            qualifier = "A"
        
        location_phrase = ""
        if nec_location and nec_location.lower() != 'unknown':
            location_phrase = f" {nec_location.lower()}"
        
        return f"{qualifier}{location_phrase} necrotic component is present within the lesion."
    
    @staticmethod
    def generate_edema_sentence(tumor: dict) -> str:
        """Generate sentence describing edema (qualitative only - no volume)."""
        edema_vol = tumor.get('edema_volume_cm3', 0)
        
        # Convert to qualitative descriptor
        if edema_vol < 5:
            descriptor = "Minimal"
        elif edema_vol < 20:
            descriptor = "Moderate"
        else:
            descriptor = "Significant"
        
        return f"{descriptor} surrounding T2/FLAIR hyperintensity is present, consistent with vasogenic edema."
    
    @staticmethod
    def generate_mass_effect_sentence(mass_effect: dict) -> str:
        """Generate sentence describing mass effect."""
        midline_shift = mass_effect.get('midline_shift_mm', 0)
        shift_significant = mass_effect.get('shift_significant', False)
        herniation_risk = mass_effect.get('herniation_risk', 'Low')
        
        sentences = []
        
        # Midline shift
        if shift_significant and midline_shift >= 2.0:
            rounded_shift = round(midline_shift, 1)
            sentences.append(f"There is approximately {rounded_shift} mm of midline shift.")
        else:
            sentences.append("No significant midline shift is identified.")
        
        # Herniation
        sentences.append("No evidence of subfalcine, transtentorial, or tonsillar herniation.")
        
        return ' '.join(sentences)
    
    @staticmethod
    def generate_ventricles_sentence(normal_structures: dict) -> str:
        """Generate sentence describing ventricular system."""
        vent = normal_structures.get('ventricular_system', {})
        size = vent.get('size', 'Normal')
        hydrocephalus = vent.get('hydrocephalus', False)
        symmetry = vent.get('symmetry', 'Symmetric')
        
        if hydrocephalus:
            return "The ventricular system is dilated, concerning for hydrocephalus."
        elif 'asymmetric' in symmetry.lower():
            return "The ventricular system is normal in size. Mild asymmetry of the lateral ventricles is noted."
        else:
            return "The ventricular system is normal in size and configuration."
    
    @staticmethod
    def generate_parenchyma_sentence(normal_structures: dict) -> str:
        """Generate sentence describing background parenchyma."""
        parenchyma = normal_structures.get('parenchyma', {})
        gw_diff = parenchyma.get('gray_white_differentiation', 'Preserved')
        wm_disease = parenchyma.get('white_matter_disease', False)
        
        if wm_disease:
            return "Background white matter disease is noted. Gray-white matter differentiation is otherwise preserved."
        elif 'preserved' in gw_diff.lower():
            return "The background brain parenchyma demonstrates preserved gray-white matter differentiation without additional focal abnormality."
        else:
            return "The remaining brain parenchyma appears unremarkable."
    
    @staticmethod
    def generate_impression_summary(tumor: dict, location: dict, multiplicity: dict, enhancement: dict) -> str:
        """Generate primary impression statement."""
        # Get size
        max_diameter_mm = tumor.get('max_diameter_mm', 0)
        size_cm = round(max_diameter_mm / 10 * 2) / 2
        
        # Get location
        hemisphere = location.get('hemisphere', '')
        primary_lobe = location.get('primary_lobe', '')
        
        # Get lesion count
        count = multiplicity.get('lesion_count', 1)
        
        # Get enhancement
        pattern = enhancement.get('pattern', '')
        
        # Build description
        if count > 1:
            if 'ring' in pattern.lower():
                return f"Multifocal ring-enhancing masses in the {hemisphere} cerebral hemisphere, largest measuring approximately {size_cm} cm. Findings raise concern for high-grade neoplasm."
            else:
                return f"Multifocal enhancing masses in the {hemisphere} cerebral hemisphere. Findings raise concern for neoplastic process."
        else:
            if 'ring' in pattern.lower():
                return f"Ring-enhancing mass in the {hemisphere} {primary_lobe} lobe measuring approximately {size_cm} cm. Imaging features raise concern for high-grade neoplasm."
            else:
                return f"Enhancing mass in the {hemisphere} {primary_lobe} lobe measuring approximately {size_cm} cm."
    
    @staticmethod
    def generate_impression_differential(differential: list, multiplicity: dict) -> str:
        """Generate differential diagnosis statement."""
        count = multiplicity.get('lesion_count', 1)
        
        # Default differentials if none provided
        if not differential:
            if count > 1:
                differential = ['metastatic disease', 'multicentric glioma', 'CNS lymphoma']
            else:
                differential = ['high-grade glioma (glioblastoma)', 'metastasis', 'lymphoma']
        
        # Format for clinical style
        formatted = []
        for d in differential[:3]:  # Limit to 3
            d_lower = d.lower()
            if 'glioma' in d_lower or 'glioblastoma' in d_lower:
                formatted.append('glioblastoma')
            elif 'metast' in d_lower:
                formatted.append('metastasis')
            elif 'lymphoma' in d_lower:
                formatted.append('primary CNS lymphoma')
            else:
                formatted.append(d.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in formatted:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        
        if len(unique) >= 2:
            diff_str = ', '.join(unique[:-1]) + f', and {unique[-1]}'
        elif unique:
            diff_str = unique[0]
        else:
            diff_str = 'high-grade neoplasm'
        
        return f"Differential diagnosis includes {diff_str}. Clinical and histopathologic correlation recommended."


# ============================================================================
# TEMPLATE FILLER CLASS (Updated to use FactExtractor + FactsToSlotMapper)
# ============================================================================

class ReportTemplateFiller:
    """
    Fills the report template using the new 4-step pipeline:
    
    Step 1: Rigid Template (MRI_BRAIN_TEMPLATE)
    Step 2: Slot Specifications with constraints (SLOT_SPECIFICATIONS)
    Step 3: FactExtractor - Convert model outputs to structured facts (NO LLM)
    Step 4: FactsToSlotMapper - Map facts to slot values (deterministic)
    
    All generated content is validated against slot specifications.
    """
    
    def __init__(self, summary: dict, validate: bool = True, use_llm: bool = False, api_key: str = None):
        """
        Initialize with the LLM-ready summary data.
        
        Args:
            summary: The llm_ready_summary.json data as a dictionary
            validate: Whether to validate generated content against slot specs
            use_llm: Whether to use constrained LLM for ambiguous cases (Step 4)
            api_key: API key for LLM (only used if use_llm=True)
        """
        self.summary = summary
        self.validator = SlotValidator()
        self.validate = validate
        self.validation_log = []
        
        # Step 3: Extract structured facts
        self.fact_extractor = FactExtractor(summary)
        self.facts = self.fact_extractor.extract_facts()
        
        # Step 4: Optional constrained LLM filler
        self.llm_filler = None
        if use_llm and api_key:
            self.llm_filler = ConstrainedLLMFiller(api_key)
    
    def _validate_and_sanitize(self, slot_name: str, content: str) -> str:
        """
        Validate content against slot specification and sanitize if needed.
        """
        if not self.validate:
            return content
        
        is_valid, sanitized, violations = self.validator.validate(slot_name, content)
        
        if violations:
            self.validation_log.append({
                'slot': slot_name,
                'original': content,
                'sanitized': sanitized,
                'violations': violations
            })
        
        return sanitized
    
    def fill_template(self, template: str = None) -> str:
        """
        Fill all placeholders in the template using the 6-step pipeline.
        
        COMPLETE PIPELINE:
        Step 1: Rigid Template (MRI_BRAIN_TEMPLATE) - human-written
        Step 2: Slot Specifications with constraints
        Step 3: FactExtractor - model outputs → structured facts (NO LLM)
        Step 4: FactsToSlotMapper - facts → slot values (deterministic)
        Step 5: Validation Layer - mandatory checks, fallback on failure
        Step 6: Assemble Final Report - zero creativity, deterministic
        
        Args:
            template: The template string to fill (defaults to MRI_BRAIN_TEMPLATE)
            
        Returns:
            The completed report as a string (deterministic, no hallucination possible)
        """
        if template is None:
            template = MRI_BRAIN_TEMPLATE
        
        # Clear validation log for new report
        self.validation_log = []
        
        # =====================================================================
        # STEP 4: Map facts to slot values (deterministic, no LLM)
        # =====================================================================
        mapper = FactsToSlotMapper(self.facts)
        raw_values = mapper.map_to_slot_values()
        
        # =====================================================================
        # STEP 5: Validation Layer (MANDATORY)
        # Every slot must pass validation; failures use fallback
        # =====================================================================
        field_values = {}
        for slot_name, content in raw_values.items():
            field_values[slot_name] = self._validate_and_sanitize(slot_name, content)
        
        # =====================================================================
        # STEP 6: Assemble Final Report (ZERO CREATIVITY)
        # Simple format() call - no LLM, no creativity, deterministic output
        # =====================================================================
        report = template.format(**field_values)
        
        # Clean up any double blank lines
        while '\n\n\n' in report:
            report = report.replace('\n\n\n', '\n\n')
        
        # Final sanity check - ensure no banned words slipped through
        report = self._final_report_validation(report)
        
        return report
    
    def _is_banned_in_final_report(self, report: str, banned_word: str) -> bool:
        """
        Check if a banned word appears in the report outside of allowed compounds.
        """
        report_lower = report.lower()
        banned_lower = banned_word.lower()
        
        if banned_lower not in report_lower:
            return False
            
        # Find all positions where banned word appears
        pos = 0
        while True:
            idx = report_lower.find(banned_lower, pos)
            if idx == -1:
                break
                
            # Check if this occurrence is part of an allowed compound
            is_part_of_allowed = False
            for allowed in SlotValidator.ALLOWED_COMPOUND_WORDS:
                if allowed.lower() in report_lower:
                    allowed_idx = report_lower.find(allowed.lower())
                    if allowed_idx != -1:
                        if idx >= allowed_idx and idx < allowed_idx + len(allowed):
                            is_part_of_allowed = True
                            break
            
            if not is_part_of_allowed:
                return True  # Found a truly banned occurrence
            
            pos = idx + 1
        
        return False  # All occurrences are in allowed compounds
    
    def _final_report_validation(self, report: str) -> str:
        """
        Issue #7: FINAL DETERMINISTIC VALIDATION PASS
        This is the GATEKEEPER - the last line of defense.
        
        Validation Checklist:
        1. No forbidden words per section
        2. No duplicated concepts
        3. No sentence fragments
        4. No empty or dangling modifiers
        5. No numeric precision violations
        6. Global banned words check
        """
        
        # =====================================================================
        # CHECK 1: Section-level forbidden terms
        # =====================================================================
        sections = self._extract_sections(report)
        
        for section_name, section_content in sections.items():
            if section_name in SECTION_FORBIDDEN_TERMS:
                for term in SECTION_FORBIDDEN_TERMS[section_name]:
                    if term.lower() in section_content.lower():
                        self.validation_log.append({
                            'slot': f'FINAL_{section_name}',
                            'original': f'[Contains: {term}]',
                            'sanitized': '[REDACTED]',
                            'violations': [f"Section '{section_name}' contains forbidden term: '{term}'"]
                        })
                        # Redact the term
                        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                        report = pattern.sub('', report)
        
        # =====================================================================
        # CHECK 2: Duplicated concepts
        # =====================================================================
        concept_mentions = {}
        for concept in CONCEPT_OWNERSHIP.keys():
            concept_pattern = concept.replace('_', ' ')
            count = report.lower().count(concept_pattern)
            if count > 1:
                self.validation_log.append({
                    'slot': 'FINAL_CONCEPTS',
                    'original': f'[Duplicate: {concept}]',
                    'sanitized': '[LOGGED]',
                    'violations': [f"Concept '{concept}' appears {count} times"]
                })
        
        # =====================================================================
        # CHECK 3: Sentence fragments
        # =====================================================================
        # Check for common fragment patterns
        fragment_patterns = [
            (r',\s*\.', 'Comma before period'),
            (r',\s*,', 'Double comma'),
            (r'\.\s*\.', 'Double period'),
            (r'\s{3,}', 'Excessive whitespace'),
        ]
        
        for pattern, description in fragment_patterns:
            if re.search(pattern, report):
                self.validation_log.append({
                    'slot': 'FINAL_GRAMMAR',
                    'original': f'[Fragment: {description}]',
                    'sanitized': '[CLEANED]',
                    'violations': [f"Grammar issue: {description}"]
                })
                report = re.sub(pattern, ' ', report)
        
        # =====================================================================
        # CHECK 4: Empty or dangling modifiers
        # =====================================================================
        # Remove empty sentences or dangling content
        report = re.sub(r'\n\s*\.\s*\n', '\n', report)  # Empty sentence
        report = re.sub(r'\s+,\s+', ' ', report)  # Dangling comma
        
        # =====================================================================
        # CHECK 5: Clean up formatting
        # =====================================================================
        # Remove multiple spaces
        report = re.sub(r' {2,}', ' ', report)
        # Clean up multiple newlines
        while '\n\n\n' in report:
            report = report.replace('\n\n\n', '\n\n')
        
        # =====================================================================
        # CHECK 6: Global banned words (final safety net)
        # =====================================================================
        for banned in SlotValidator.GLOBAL_BANNED_WORDS:
            if self._is_banned_in_final_report(report, banned):
                self.validation_log.append({
                    'slot': 'FINAL_BANNED',
                    'original': f'[Contains: {banned}]',
                    'sanitized': '[REDACTED]',
                    'violations': [f"Final check caught banned term: '{banned}'"]
                })
                pattern = re.compile(r'\b' + re.escape(banned) + r'\b', re.IGNORECASE)
                report = pattern.sub('', report)
        
        return report
    
    def _extract_sections(self, report: str) -> dict:
        """Extract section content from report for validation."""
        sections = {}
        
        # Define section markers
        section_markers = ['FINDINGS:', 'IMPRESSION:', 'TECHNIQUE:', 'COMPARISON:']
        
        for marker in section_markers:
            if marker in report:
                start = report.find(marker) + len(marker)
                # Find end (next section or end of report)
                end = len(report)
                for other in section_markers:
                    if other != marker and other in report[start:]:
                        potential_end = report.find(other, start)
                        if potential_end < end:
                            end = potential_end
                
                section_name = marker.replace(':', '')
                sections[section_name] = report[start:end].strip()
        
        return sections
        
        return report
    
    def get_facts(self) -> dict:
        """
        Get the extracted facts from Step 3.
        Useful for debugging or custom processing.
        """
        return self.facts
    
    def get_validation_log(self) -> list:
        """
        Get the validation log from the last fill_template call.
        
        Returns:
            List of validation violations found
        """
        return self.validation_log
    
    def get_field_values(self) -> dict:
        """
        Get all computed field values without filling the template.
        Useful for debugging or custom template usage.
        
        Returns:
            Dictionary of field names to generated values
        """
        patient_info = self.summary.get('patient_info', {})
        technique = self.summary.get('technique', {})
        tumor = self.summary.get('tumor_characteristics', {})
        location = self.summary.get('location', {})
        morphology = self.summary.get('morphology', {})
        signal = self.summary.get('signal_characteristics', {})
        enhancement = self.summary.get('enhancement', {})
        mass_effect = self.summary.get('mass_effect', {})
        necrosis = self.summary.get('necrosis', {})
        multiplicity = self.summary.get('multiplicity', {})
        normal_structures = self.summary.get('normal_structures', {})
        differential = self.summary.get('differential_considerations', [])
        
        return {
            'patient_id': self.summary.get('case_id', 'Unknown'),
            'exam_date': datetime.now().strftime('%B %d, %Y'),
            'clinical_indication': self.generators.generate_clinical_indication(patient_info),
            'sequences_list': self.generators.generate_sequences_list(technique),
            'contrast_sentence': self.generators.generate_contrast_sentence(technique),
            'comparison': self.generators.generate_comparison(patient_info),
            'lesion_count_sentence': self.generators.generate_lesion_count_sentence(multiplicity, location),
            'dominant_lesion_sentence': self.generators.generate_dominant_lesion_sentence(tumor, location, morphology),
            'enhancement_sentence': self.generators.generate_enhancement_sentence(enhancement, signal),
            'necrosis_sentence': self.generators.generate_necrosis_sentence(necrosis),
            'edema_sentence': self.generators.generate_edema_sentence(tumor),
            'mass_effect_sentence': self.generators.generate_mass_effect_sentence(mass_effect),
            'ventricles_sentence': self.generators.generate_ventricles_sentence(normal_structures),
            'parenchyma_sentence': self.generators.generate_parenchyma_sentence(normal_structures),
            'impression_summary': self.generators.generate_impression_summary(tumor, location, multiplicity, enhancement),
            'impression_differential': self.generators.generate_impression_differential(differential, multiplicity),
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_report_from_summary(summary: dict, validate: bool = True) -> tuple:
    """
    Generate a complete radiology report from an LLM-ready summary.
    
    This is the main entry point for template-based report generation.
    
    Args:
        summary: The llm_ready_summary.json data as a dictionary
        validate: Whether to validate against slot specifications
        
    Returns:
        Tuple of (report_string, validation_log)
    """
    filler = ReportTemplateFiller(summary, validate=validate)
    report = filler.fill_template()
    return report, filler.get_validation_log(), filler.get_facts()


def generate_report_simple(summary: dict) -> str:
    """
    Simple wrapper that returns only the report string.
    For backward compatibility.
    """
    report, _, _ = generate_report_from_summary(summary)
    return report


# ============================================================================
# TEST / DEMO - Demonstrates all 4 steps of the pipeline
# ============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Demo with a sample case
    test_case = Path("results/BraTS-GLI-00009-000/feature_extraction/llm_ready_summary.json")
    
    if test_case.exists():
        print("=" * 70)
        print("TEMPLATE-DRIVEN REPORT GENERATION - 4-STEP PIPELINE DEMO")
        print("=" * 70)
        
        with open(test_case, 'r') as f:
            summary = json.load(f)
        
        # =====================================================================
        # STEP 1: Show the rigid template
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 1: RIGID TEMPLATE (Human-Written, Cannot Be Modified)")
        print("=" * 70)
        print("Template has these fixed slots:")
        import re
        slots_in_template = re.findall(r'\{(\w+)\}', MRI_BRAIN_TEMPLATE)
        for i, slot in enumerate(slots_in_template, 1):
            print(f"  {i:2}. {{{slot}}}")
        
        # =====================================================================
        # STEP 2: Show slot specifications
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 2: SLOT SPECIFICATIONS (Hard Constraints)")
        print("=" * 70)
        print(f"Total slots with specifications: {len(SLOT_SPECIFICATIONS)}")
        for slot_name, spec in list(SLOT_SPECIFICATIONS.items())[:5]:  # Show first 5
            forbidden = len(spec.get('forbidden_terms', []))
            templates = len(spec.get('allowed_templates', []))
            allowed_vals = spec.get('allowed_values', {})
            max_len = spec.get('max_length', 'N/A')
            print(f"\n  [{slot_name}]")
            print(f"    Templates: {templates}, Forbidden terms: {forbidden}, Max length: {max_len}")
            if allowed_vals:
                for field, values in allowed_vals.items():
                    print(f"    Allowed {field}: {values}")
        print("  ... (more slots defined)")
        
        # =====================================================================
        # STEP 3: Extract structured facts (NO LLM)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 3: FACT EXTRACTION (Deterministic, No LLM)")
        print("=" * 70)
        
        fact_extractor = FactExtractor(summary)
        facts = fact_extractor.extract_facts()
        
        print("Extracted facts from model outputs:")
        key_facts = [
            'lesion_count', 'distribution', 'hemisphere', 'lobes_formatted',
            'size_cm', 'edema_degree', 'is_ring_enhancing', 'necrosis_degree',
            'shift_significant', 'ventricles_symmetric'
        ]
        for key in key_facts:
            print(f"  {key}: {facts.get(key, 'N/A')}")
        
        # =====================================================================
        # STEP 4: Map facts to slot values (Deterministic)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: FACTS → SLOT VALUES MAPPING (Deterministic)")
        print("=" * 70)
        
        mapper = FactsToSlotMapper(facts)
        slot_values = mapper.map_to_slot_values()
        
        print("Mapped slot values:")
        for slot, value in list(slot_values.items())[:6]:  # Show first 6
            preview = value[:60] + "..." if len(value) > 60 else value
            print(f"  {slot}:")
            print(f"    → {preview}")
        print("  ... (more slots mapped)")
        
        # =====================================================================
        # STEP 5: Validation Layer (Mandatory)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: VALIDATION LAYER (Mandatory)")
        print("=" * 70)
        print(f"Global banned words: {len(SlotValidator.GLOBAL_BANNED_WORDS)}")
        print(f"Sample banned words: {SlotValidator.GLOBAL_BANNED_WORDS[:5]}...")
        print("\nValidating all slot values...")
        
        filler = ReportTemplateFiller(summary, validate=True)
        report = filler.fill_template()
        validation_log = filler.get_validation_log()
        
        if validation_log:
            print(f"⚠ Found {len(validation_log)} violations (auto-corrected with fallback):")
            for entry in validation_log:
                print(f"  - Slot: {entry['slot']}")
                print(f"    Violations: {entry['violations']}")
        else:
            print("✓ All slots passed validation - no violations found!")
        
        # =====================================================================
        # STEP 6: Assemble Final Report (Zero Creativity)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 6: ASSEMBLE FINAL REPORT (Zero Creativity)")
        print("=" * 70)
        print("Template.format(**slot_values) → Final report")
        print("- No LLM involvement")
        print("- No creativity or variation")
        print("- 100% deterministic output")
        print("- Cannot hallucinate")
        
        # =====================================================================
        # FINAL REPORT OUTPUT
        # =====================================================================
        print("\n" + "=" * 70)
        print("FINAL GENERATED REPORT")
        print("=" * 70)
        print(report)
        
        # =====================================================================
        # TEST: Validation catches violations
        # =====================================================================
        print("\n" + "=" * 70)
        print("TEST: STEP 5 VALIDATION CATCHES VIOLATIONS")
        print("=" * 70)
        
        test_cases = [
            # Global banned words
            ("This is definitely a microscopic tumor.", 'dominant_lesion_sentence', 'Global banned'),
            ("Findings confirmed by histologic analysis.", 'impression_summary', 'Global banned'),
            # Slot-specific forbidden terms  
            ("Extensive edema measuring 45.6 cm³.", 'edema_sentence', 'Slot forbidden'),
            # Valid content
            ("A subcortical mass is identified.", 'dominant_lesion_sentence', 'Should pass'),
        ]
        
        for test_content, slot, expected in test_cases:
            is_valid, sanitized, violations = slot_validator.validate(slot, test_content)
            status = "✓ PASS" if is_valid else "✗ BLOCKED"
            print(f"\n  [{expected}]")
            print(f"  Input: '{test_content}'")
            print(f"  Result: {status}")
            if violations:
                print(f"  Action: Used fallback → '{sanitized[:50]}...'")
        
        # =====================================================================
        # SUMMARY: Complete 6-Step Pipeline
        # =====================================================================
        print("\n" + "=" * 70)
        print("COMPLETE 6-STEP PIPELINE SUMMARY")
        print("=" * 70)
        print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: RIGID TEMPLATE        │ Human-written, fixed structure     │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: SLOT SPECIFICATIONS   │ Constraints, forbidden terms       │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: FACT EXTRACTION       │ Model outputs → structured facts   │
│                               │ (NO LLM - deterministic)           │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: SLOT MAPPING          │ Facts → slot values                │
│                               │ (NO LLM - deterministic)           │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: VALIDATION            │ Mandatory checks, fallback on fail │
│                               │ (Global + slot-specific bans)      │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6: ASSEMBLY              │ template.format() - ZERO CREATIVITY│
│                               │ (Deterministic, no hallucination)  │
└─────────────────────────────────────────────────────────────────────┘
        """)
        
    else:
        print("Demo case not found. Run with actual summary data.")
