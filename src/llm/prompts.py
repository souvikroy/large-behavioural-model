"""Prompt templates for LLM tasks."""

import json
from typing import List, Dict, Any


def get_concept_extraction_prompt(question_text: str) -> str:
    """
    Get prompt for concept extraction from question text.
    
    Args:
        question_text: Question text to analyze
        
    Returns:
        Formatted prompt
    """
    return f"""Extract educational concepts from the following question text. 
Return only a JSON array of concept names, one per line. Focus on mathematical, scientific, or educational concepts.

Question: {question_text}

Concepts (JSON array):"""


def get_misconception_detection_prompt(question_text: str) -> str:
    """
    Get prompt for misconception detection.
    
    Args:
        question_text: Question text to analyze
        
    Returns:
        Formatted prompt
    """
    return f"""Analyze the following question to detect if it tests for misconceptions or common errors.
Return a JSON object with these keys:
- "has_misconception_indicators": boolean
- "misconception_type": string (e.g., "conceptual", "procedural", "none")
- "indicators": array of strings describing detected indicators

Question: {question_text}

Analysis (JSON):"""


def get_question_analysis_prompt(question_text: str) -> str:
    """
    Get prompt for comprehensive question analysis.
    
    Args:
        question_text: Question text to analyze
        
    Returns:
        Formatted prompt
    """
    return f"""Analyze the following educational question and provide:
1. Main concepts being tested
2. Difficulty level indicators
3. Question type (multiple choice, open-ended, etc.)
4. Cognitive level (remember, understand, apply, analyze, evaluate, create)

Return as JSON object.

Question: {question_text}

Analysis (JSON):"""


def get_prerequisite_detection_prompt(concept: str, related_concepts: List[str]) -> str:
    """
    Get prompt for prerequisite relationship detection.
    
    Args:
        concept: Target concept
        related_concepts: List of potentially related concepts
        
    Returns:
        Formatted prompt
    """
    concepts_str = ", ".join(related_concepts[:10])  # Limit to avoid too long prompts
    
    return f"""Identify which concepts are prerequisites for learning "{concept}".
Consider the following related concepts: {concepts_str}

Return a JSON object with:
- "prerequisites": array of concept names that are prerequisites
- "confidence": float between 0 and 1

Analysis (JSON):"""


def get_gap_analysis_prompt(gaps: Dict[str, Any]) -> str:
    """
    Get prompt for gap analysis and remediation strategy.
    
    Args:
        gaps: Dictionary with gap analysis data
        
    Returns:
        Formatted prompt
    """
    gaps_summary = json.dumps(gaps, indent=2)
    
    return f"""Based on the following student competency gap analysis, generate a remediation strategy.

Gap Analysis:
{gaps_summary}

Return a JSON object with:
- "priority_areas": array of areas to focus on
- "recommended_actions": array of action descriptions
- "prerequisite_review": array of prerequisites to review

Remediation Strategy (JSON):"""


def get_misconception_remediation_prompt(misconceptions: List[Dict[str, Any]]) -> str:
    """
    Get prompt for misconception remediation recommendations.
    
    Args:
        misconceptions: List of misconception dictionaries
        
    Returns:
        Formatted prompt
    """
    misconceptions_str = json.dumps(misconceptions, indent=2)
    
    return f"""Based on the following identified misconceptions, recommend remediation strategies.

Misconceptions:
{misconceptions_str}

Return a JSON object with:
- "questions_to_review": array of question IDs
- "prerequisites_to_review": array of prerequisite concepts
- "concepts_to_clarify": array of concepts that need clarification

Remediation (JSON):"""


def get_semantic_similarity_prompt(text1: str, text2: str) -> str:
    """
    Get prompt for semantic similarity analysis.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Formatted prompt
    """
    return f"""Rate the semantic similarity between these two educational texts on a scale of 0.0 to 1.0.
Consider: concepts covered, difficulty level, question type, learning objectives.

Text 1: {text1}

Text 2: {text2}

Return only a JSON object with:
- "similarity": float between 0.0 and 1.0
- "reasoning": brief explanation

Similarity (JSON):"""
