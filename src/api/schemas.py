"""API request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class CompetencyPredictionRequest(BaseModel):
    """Request schema for competency prediction."""
    question_id: Optional[str] = None
    question_text: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    chapter: Optional[str] = None
    learning_objective: Optional[str] = None
    learning_unit: Optional[str] = None
    bloom_tag: Optional[str] = None
    three_pl: Optional[float] = None


class CompetencyPredictionResponse(BaseModel):
    """Response schema for competency prediction."""
    predicted_competency: float = Field(..., ge=0.0, le=1.0)
    confidence: Optional[float] = None


class QuestionRecommendationRequest(BaseModel):
    """Request schema for question recommendation."""
    student_profile: Dict[str, Any]
    target_objectives: Optional[List[str]] = None
    num_recommendations: Optional[int] = 10


class QuestionRecommendationResponse(BaseModel):
    """Response schema for question recommendation."""
    recommendations: List[Dict[str, Any]]
    reasoning: Optional[str] = None


class LearningPathRequest(BaseModel):
    """Request schema for learning path generation."""
    student_profile: Dict[str, Any]
    target_concept: Optional[str] = None
    target_chapter: Optional[str] = None


class LearningPathResponse(BaseModel):
    """Response schema for learning path."""
    path: List[Dict[str, Any]]
    total_steps: int


class GapAnalysisRequest(BaseModel):
    """Request schema for gap analysis."""
    student_responses: List[Dict[str, Any]]
    competency_threshold: Optional[float] = 0.6


class GapAnalysisResponse(BaseModel):
    """Response schema for gap analysis."""
    gaps: Dict[str, Any]
    root_causes: List[Dict[str, Any]]
    remediation_strategy: Dict[str, Any]


class MisconceptionDetectionRequest(BaseModel):
    """Request schema for misconception detection."""
    student_responses: List[Dict[str, Any]]


class MisconceptionDetectionResponse(BaseModel):
    """Response schema for misconception detection."""
    misconceptions: List[Dict[str, Any]]
    total_identified: int
    remediation: Dict[str, Any]


class ConceptInfoRequest(BaseModel):
    """Request schema for concept information."""
    concept_id: str


class ConceptInfoResponse(BaseModel):
    """Response schema for concept information."""
    concept_details: Dict[str, Any]


class PrerequisitesRequest(BaseModel):
    """Request schema for prerequisites."""
    concept_id: str
    max_depth: Optional[int] = 5


class PrerequisitesResponse(BaseModel):
    """Response schema for prerequisites."""
    prerequisites: List[str]


class LearningPathQueryRequest(BaseModel):
    """Request schema for learning path query."""
    from_concept: str
    to_concept: str
    respect_prerequisites: Optional[bool] = True


class LearningPathQueryResponse(BaseModel):
    """Response schema for learning path query."""
    path: List[str]
    path_length: int
