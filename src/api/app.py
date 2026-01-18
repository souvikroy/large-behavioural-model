"""FastAPI application for behavioral model."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import yaml
from typing import Dict, List, Optional

from .schemas import (
    CompetencyPredictionRequest, CompetencyPredictionResponse,
    QuestionRecommendationRequest, QuestionRecommendationResponse,
    LearningPathRequest, LearningPathResponse,
    GapAnalysisRequest, GapAnalysisResponse,
    MisconceptionDetectionRequest, MisconceptionDetectionResponse,
    ConceptInfoRequest, ConceptInfoResponse,
    PrerequisitesRequest, PrerequisitesResponse,
    LearningPathQueryRequest, LearningPathQueryResponse
)

from ..orchestrator import Orchestrator

# Initialize FastAPI app
app = FastAPI(
    title="Behavioral Model for Adaptive Learning API",
    description="API for predicting competency, recommending questions, and analyzing learning gaps",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None
runtime_orchestrator = None


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    global orchestrator, runtime_orchestrator
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Initialize orchestrator
        orchestrator = Orchestrator()
        
        # Get data path from config
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        data_path = config.get('data', {}).get('input_file', 'question-response.json')
        
        # Initialize services
        init_result = orchestrator.initialize(data_path)
        runtime_orchestrator = orchestrator.get_runtime_orchestrator()
        
        print(f"Orchestrator initialized successfully! Services: {init_result.get('services_initialized', 0)}")
        print(f"Health status: {init_result.get('health', {}).get('overall', 'unknown')}")
    except Exception as e:
        print(f"Warning: Could not initialize orchestrator: {e}")
        print("API will run but some endpoints may not work without initialized services")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if orchestrator is None:
        return {"status": "not_initialized", "service": "behavioral-model-api"}
    
    health = orchestrator.health_check()
    return {
        "status": health.get("overall", "unknown"),
        "service": "behavioral-model-api",
        "services": health.get("services", {})
    }


@app.post("/predict/competency", response_model=CompetencyPredictionResponse)
async def predict_competency(request: CompetencyPredictionRequest):
    """
    Predict competency score for a question.
    
    Args:
        request: Competency prediction request
        
    Returns:
        Predicted competency score
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Convert request to question data dictionary
    question_data = {
        'question_id': request.question_id,
        'question_text': request.question_text,
        'Subject': request.subject,
        'grade': request.grade,
        'chapter': request.chapter,
        'learning_objective': request.learning_objective,
        'learning_unit': request.learning_unit,
        'Bloom_tag': request.bloom_tag,
        '3PL': request.three_pl
    }
    
    # Predict using orchestrator
    result = runtime_orchestrator.predict_competency(question_data)
    
    return CompetencyPredictionResponse(
        predicted_competency=result['predicted_competency'],
        confidence=result.get('confidence')
    )


@app.post("/recommend/questions", response_model=QuestionRecommendationResponse)
async def recommend_questions(request: QuestionRecommendationRequest):
    """
    Get question recommendations for a student.
    
    Args:
        request: Question recommendation request
        
    Returns:
        List of recommended questions
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    recommendations = runtime_orchestrator.recommend_questions(
        student_profile=request.student_profile,
        target_objectives=request.target_objectives,
        num_recommendations=request.num_recommendations or 10
    )
    
    return QuestionRecommendationResponse(
        recommendations=recommendations,
        reasoning=None
    )


@app.post("/generate/path", response_model=LearningPathResponse)
async def generate_learning_path(request: LearningPathRequest):
    """
    Generate learning path for a student.
    
    Args:
        request: Learning path request
        
    Returns:
        Learning path
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    path = runtime_orchestrator.generate_learning_path(
        student_profile=request.student_profile,
        target_concept=request.target_concept,
        target_chapter=request.target_chapter
    )
    
    return LearningPathResponse(
        path=path,
        total_steps=len(path)
    )


@app.post("/analyze/gaps", response_model=GapAnalysisResponse)
async def analyze_gaps(request: GapAnalysisRequest):
    """
    Analyze competency gaps.
    
    Args:
        request: Gap analysis request
        
    Returns:
        Gap analysis results
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Use orchestrator for comprehensive analysis
    analysis = runtime_orchestrator.analyze_student(
        student_responses=request.student_responses,
        competency_threshold=request.competency_threshold or 0.6
    )
    
    return GapAnalysisResponse(
        gaps=analysis['gaps'],
        root_causes=analysis['gaps'].get('root_causes', []),
        remediation_strategy=analysis['gap_remediation']
    )


@app.post("/detect/misconceptions", response_model=MisconceptionDetectionResponse)
async def detect_misconceptions(request: MisconceptionDetectionRequest):
    """
    Detect misconceptions from student responses.
    
    Args:
        request: Misconception detection request
        
    Returns:
        Detected misconceptions
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Use orchestrator for comprehensive analysis
    analysis = runtime_orchestrator.analyze_student(
        student_responses=request.student_responses,
        competency_threshold=0.3  # Lower threshold for misconception detection
    )
    
    return MisconceptionDetectionResponse(
        misconceptions=analysis['misconceptions']['misconceptions'],
        total_identified=analysis['misconceptions']['total_identified'],
        remediation=analysis['misconception_remediation']
    )


@app.get("/graph/concepts/{concept_id}", response_model=ConceptInfoResponse)
async def get_concept_info(concept_id: str):
    """
    Get concept details from knowledge graph.
    
    Args:
        concept_id: Concept identifier
        
    Returns:
        Concept details
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    result = runtime_orchestrator.get_concept_info(concept_id)
    
    return ConceptInfoResponse(concept_details=result['concept_details'])


@app.get("/graph/prerequisites/{concept_id}", response_model=PrerequisitesResponse)
async def get_prerequisites(concept_id: str, max_depth: int = 5):
    """
    Get prerequisites for a concept.
    
    Args:
        concept_id: Concept identifier
        max_depth: Maximum depth to traverse
        
    Returns:
        List of prerequisites
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    prerequisites = runtime_orchestrator.get_prerequisites(concept_id, max_depth)
    
    return PrerequisitesResponse(prerequisites=prerequisites)


@app.get("/graph/path/{from_concept}/{to_concept}", response_model=LearningPathQueryResponse)
async def get_learning_path(
    from_concept: str,
    to_concept: str,
    respect_prerequisites: bool = True
):
    """
    Find learning path between concepts.
    
    Args:
        from_concept: Starting concept
        to_concept: Target concept
        respect_prerequisites: Whether to respect prerequisites
        
    Returns:
        Learning path
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    path = runtime_orchestrator.find_learning_path_between_concepts(
        from_concept, to_concept, respect_prerequisites
    )
    
    return LearningPathQueryResponse(
        path=path,
        path_length=len(path)
    )


@app.get("/graph/misconceptions/{concept_id}")
async def get_misconceptions_for_concept(concept_id: str):
    """
    Get misconceptions associated with a concept.
    
    Args:
        concept_id: Concept identifier
        
    Returns:
        List of misconceptions
    """
    if runtime_orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    result = runtime_orchestrator.get_concept_info(concept_id)
    misconceptions = result['concept_details'].get('misconceptions', [])
    
    return {"misconceptions": misconceptions}


if __name__ == "__main__":
    import uvicorn
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    uvicorn.run(
        app,
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000)
    )
