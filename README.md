# Behavioral Model for Adaptive Learning System

A comprehensive behavioral modeling system that analyzes student performance patterns, detects misconceptions from question text, and constructs a knowledge graph to enable adaptive learning. The system helps teachers set up personalized learning materials by predicting competency, recommending questions, identifying learning gaps, and generating optimal learning paths.

## Features

- **Competency Prediction**: Predict student competency scores for questions using ensemble models (XGBoost, LightGBM, Neural Networks)
- **Misconception Detection**: Identify misconceptions from question text using NLP analysis and performance patterns
- **Knowledge Graph**: Construct and query educational concept relationships (prerequisites, misconceptions, difficulty progression)
- **Question Recommendation**: Adaptive question recommendations based on competency, prerequisites, and learning objectives
- **Learning Path Generation**: Generate optimal learning sequences respecting prerequisites and avoiding misconceptions
- **Gap Analysis**: Identify competency gaps with root cause analysis using knowledge graph traversal
- **REST API**: FastAPI service for easy integration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## Project Structure

```
Question-Quest-recommendation-main/
├── src/
│   ├── data/              # Data preprocessing and loading
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── nlp/               # NLP analysis for misconception detection
│   │   ├── question_analyzer.py
│   │   ├── concept_extractor.py
│   │   └── text_embeddings.py
│   ├── knowledge_graph/   # Knowledge graph construction and query
│   │   ├── builder.py
│   │   ├── construction.py
│   │   ├── embeddings.py
│   │   ├── query.py
│   │   └── visualization.py
│   ├── features/          # Feature engineering
│   │   └── engineering.py
│   ├── models/            # ML models
│   │   ├── competency_predictor.py
│   │   ├── difficulty_calibrator.py
│   │   ├── question_recommender.py
│   │   ├── learning_path.py
│   │   ├── gap_analyzer.py
│   │   ├── misconception_detector.py
│   │   ├── misconception_patterns.py
│   │   └── misconception_analyzer.py
│   ├── training/          # Training pipeline
│   │   ├── pipeline.py
│   │   └── evaluator.py
│   ├── api/               # FastAPI service
│   │   ├── app.py
│   │   └── schemas.py
│   └── export/            # Model export utilities
│       └── model_exporter.py
├── models/                # Saved model files
├── knowledge_graphs/      # Saved knowledge graph files
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── config.yaml           # Configuration file
└── requirements.txt      # Python dependencies
```

## System Architecture

### Overview

The system follows a modular, service-oriented architecture with clear separation of concerns. The architecture consists of three main layers:

1. **Data Layer**: Handles data loading, preprocessing, and feature engineering
2. **Model Layer**: Contains ML models for prediction, recommendation, and analysis
3. **Orchestration Layer**: Coordinates services and manages workflows

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   REST API   │  │  Health Check│  │   CORS       │          │
│  └──────┬───────┘  └──────────────┘  └──────────────┘          │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│         │         Orchestration Layer                            │
│  ┌──────▼──────────────────────────────────────────────┐        │
│  │           Orchestrator                               │        │
│  │  ┌──────────────────┐  ┌─────────────────────────┐ │        │
│  │  │ Service Manager  │  │ Runtime Orchestrator    │ │        │
│  │  │ - Dependency Mgmt│  │ - API Workflows         │ │        │
│  │  │ - Lifecycle      │  │ - Request Coordination  │ │        │
│  │  └──────────────────┘  └─────────────────────────┘ │        │
│  │  ┌──────────────────┐                                │        │
│  │  │Training Orchestr │                                │        │
│  │  │ - Pipeline Mgmt  │                                │        │
│  │  │ - Model Training │                                │        │
│  │  └──────────────────┘                                │        │
│  └──────────────────────────────────────────────────────┘        │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│         │         Service Layer                                  │
│  ┌──────▼──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Data Services   │  │ NLP Services │  │ Knowledge Graph  │  │
│  │ - Loader        │  │ - Analyzer   │  │ - Constructor    │  │
│  │ - Preprocessor  │  │ - Extractor  │  │ - Query Engine   │  │
│  │ - Feature Eng.  │  │ - Embeddings │  │ - Visualizer     │  │
│  └─────────────────┘  └──────────────┘  └──────────────────┘  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ ML Models        │  │ Analysis Services                  │  │
│  │ - Competency Pred│  │ - Gap Analyzer                     │  │
│  │ - Difficulty Cal │  │ - Misconception Detector           │  │
│  │ - Recommender    │  │ - Misconception Analyzer          │  │
│  │ - Path Generator │  │                                   │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ Training        │  │ External Services                 │  │
│  │ - Pipeline      │  │ - OpenAI Client (optional)         │  │
│  │ - Evaluator     │  │ - Model Exporter                  │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Service Initialization Flow

Services are initialized in dependency order:

1. **Foundation Services** (no dependencies):
   - `openai_client` - Optional OpenAI integration
   - `data_loader` - Loads raw data
   - `misconception_detector` - Detects misconceptions
   - `competency_predictor` - ML model for predictions
   - `difficulty_calibrator` - Calibrates question difficulty

2. **Data Processing Services**:
   - `data_preprocessor` → depends on `data_loader`
   - `feature_engineer` → depends on `data_preprocessor`

3. **Knowledge Graph Services**:
   - `knowledge_graph_constructor` → depends on `data_preprocessor`, `misconception_detector`
   - `knowledge_graph_query` → depends on `knowledge_graph_constructor`

4. **High-Level Services** (depend on multiple services):
   - `question_recommender` → depends on `competency_predictor`, `difficulty_calibrator`, `knowledge_graph_query`
   - `learning_path_generator` → depends on `competency_predictor`, `knowledge_graph_query`
   - `gap_analyzer` → depends on `competency_predictor`, `knowledge_graph_query`
   - `misconception_analyzer` → depends on `misconception_detector`, `knowledge_graph_query`

### Data Flow

1. **Training Pipeline**:
   ```
   Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Export
   ```

2. **Runtime Prediction**:
   ```
   API Request → Runtime Orchestrator → Service Manager → Model → Response
   ```

3. **Knowledge Graph Construction**:
   ```
   Processed Data + Misconceptions → Graph Construction → Embeddings → Query Engine
   ```

### Key Design Patterns

- **Service Manager Pattern**: Centralized service lifecycle management
- **Orchestrator Pattern**: Coordinates complex workflows
- **Dependency Injection**: Services receive dependencies via constructor
- **Factory Pattern**: Service creation handled by ServiceManager
- **Strategy Pattern**: Multiple algorithms for models (XGBoost, LightGBM, Neural Networks)

## Quick Start

### 1. Data Preprocessing

```python
from src.data.preprocessing import DataPreprocessor
import pandas as pd

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data("question-response.json")
df_processed = preprocessor.preprocess(df)

# Split data
train_df, val_df, test_df = preprocessor.split_data(df_processed)
```

### 2. Build Knowledge Graph

```python
from src.knowledge_graph.construction import KnowledgeGraphConstructor
from src.models.misconception_detector import MisconceptionDetector

# Detect misconceptions
misconception_detector = MisconceptionDetector()
df_with_misconceptions = misconception_detector.detect_misconceptions_from_data(df_processed)
misconceptions = misconception_detector.catalog_misconceptions(df_with_misconceptions)

# Build knowledge graph
kg_constructor = KnowledgeGraphConstructor()
knowledge_graph = kg_constructor.construct(df_processed, misconception_detector)
```

### 3. Train Models

```python
from src.training.pipeline import TrainingPipeline

# Initialize training pipeline
pipeline = TrainingPipeline()

# Train all models
results = pipeline.train_all(df_processed, tune_hyperparameters=False)

print(f"Competency Predictor RMSE: {results['test_metrics']['competency_predictor']['rmse']:.4f}")
```

### 4. Use Models

```python
from src.models.competency_predictor import CompetencyPredictor
from src.models.question_recommender import QuestionRecommender

# Load trained model
predictor = CompetencyPredictor()
predictor.load("models/competency_predictor.pkl")

# Predict competency
predictions = predictor.predict(test_df)

# Recommend questions
recommender = QuestionRecommender(predictor, difficulty_calibrator, knowledge_graph_query)
recommender.set_question_pool(df_processed)

student_profile = {
    'avg_competency': 0.6,
    'competency_by_subject': {'Mathematics': 0.5, 'Biology': 0.7},
    'weak_areas': {'Mathematics': ['Fractions', 'Integers']}
}

recommendations = recommender.recommend(student_profile, num_recommendations=10)
```

## API Usage

### Start the API Server

```bash
cd src/api
python app.py
```

The API will be available at `http://localhost:9010` (default port from config). Interactive documentation is available at `http://localhost:9010/docs`.

### API Endpoints

#### 1. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "service": "behavioral-model-api",
  "services": {
    "data_loader": "healthy",
    "competency_predictor": "healthy",
    "knowledge_graph_query": "healthy",
    ...
  }
}
```

#### 2. Predict Competency

**Endpoint**: `POST /predict/competency`

**Request**:
```json
{
  "question_id": "Q12345",
  "question_text": "Solve for x: 2x + 5 = 13",
    "subject": "Mathematics",
    "grade": "6",
  "chapter": "Linear Equations",
  "learning_objective": "Solve one-step linear equations",
  "learning_unit": "Algebra Basics",
  "bloom_tag": "Apply",
  "three_pl": 0.65
}
```

**Response**:
```json
{
  "predicted_competency": 0.72,
  "confidence": 0.85
}
```

#### 3. Get Question Recommendations

**Endpoint**: `POST /recommend/questions`

**Request**:
```json
{
    "student_profile": {
    "avg_competency": 0.65,
    "competency_by_subject": {
      "Mathematics": 0.58,
      "Science": 0.72,
      "English": 0.65
    },
    "competency_by_chapter": {
      "Fractions": 0.45,
      "Decimals": 0.62,
      "Percentages": 0.70
    },
    "weak_areas": {
      "Mathematics": ["Fractions", "Ratio and Proportion"]
    },
    "mastered_concepts": ["basic_arithmetic", "whole_numbers"],
    "recent_topics": ["Decimals", "Percentages"]
  },
  "target_objectives": [
    "Understand fraction operations",
    "Apply fraction concepts to word problems"
  ],
    "num_recommendations": 10
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "question_id": "Q67890",
      "question_text": "What is 3/4 + 1/2?",
      "subject": "Mathematics",
      "chapter": "Fractions",
      "learning_objective": "Add fractions with different denominators",
      "bloom_tag": "Apply",
      "predicted_competency": 0.68,
      "difficulty": 0.55,
      "score": 0.82,
      "reasoning": "Matches weak area (Fractions) and target objective",
      "prerequisites_met": true,
      "estimated_time_minutes": 5
    },
    {
      "question_id": "Q67891",
      "question_text": "John has 2/3 of a pizza. He eats 1/4 of it. How much pizza is left?",
      "subject": "Mathematics",
      "chapter": "Fractions",
      "learning_objective": "Apply fraction concepts to word problems",
      "bloom_tag": "Apply",
      "predicted_competency": 0.65,
      "difficulty": 0.62,
      "score": 0.79,
      "reasoning": "Word problem addressing target objective",
      "prerequisites_met": true,
      "estimated_time_minutes": 8
    }
  ],
  "reasoning": "Recommended questions focus on Fractions chapter to address competency gaps while respecting prerequisites"
}
```

#### 4. Generate Learning Path

**Endpoint**: `POST /generate/path`

**Request**:
```json
{
    "student_profile": {
    "competency_by_chapter": {
      "Fractions": 0.40,
      "Decimals": 0.65,
      "Percentages": 0.70,
      "Algebra": 0.30
    },
    "competency_by_subject": {
      "Mathematics": 0.51
    },
    "mastered_concepts": ["basic_arithmetic", "whole_numbers", "decimals_basic"],
    "weak_areas": {
      "Mathematics": ["Fractions", "Algebra"]
    }
  },
  "target_concept": "linear_equations",
    "target_chapter": "Algebra"
}
```

**Response**:
```json
{
  "path": [
    {
      "step": 1,
      "concept_id": "fractions_basic",
      "concept_name": "Basic Fraction Operations",
      "chapter": "Fractions",
      "learning_objectives": [
        "Understand fraction representation",
        "Add and subtract fractions with same denominator"
      ],
      "prerequisites": [],
      "estimated_competency": 0.40,
      "target_competency": 0.70,
      "recommended_questions": 5,
      "estimated_time_hours": 2
    },
    {
      "step": 2,
      "concept_id": "fractions_advanced",
      "concept_name": "Advanced Fraction Operations",
      "chapter": "Fractions",
      "learning_objectives": [
        "Add and subtract fractions with different denominators",
        "Multiply and divide fractions"
      ],
      "prerequisites": ["fractions_basic"],
      "estimated_competency": 0.50,
      "target_competency": 0.75,
      "recommended_questions": 8,
      "estimated_time_hours": 3
    },
    {
      "step": 3,
      "concept_id": "algebra_intro",
      "concept_name": "Introduction to Algebra",
      "chapter": "Algebra",
      "learning_objectives": [
        "Understand variables and expressions",
        "Evaluate algebraic expressions"
      ],
      "prerequisites": ["fractions_advanced"],
      "estimated_competency": 0.30,
      "target_competency": 0.65,
      "recommended_questions": 6,
      "estimated_time_hours": 2.5
    },
    {
      "step": 4,
      "concept_id": "linear_equations",
      "concept_name": "Linear Equations",
      "chapter": "Algebra",
      "learning_objectives": [
        "Solve one-step linear equations",
        "Solve multi-step linear equations"
      ],
      "prerequisites": ["algebra_intro"],
      "estimated_competency": 0.35,
      "target_competency": 0.70,
      "recommended_questions": 10,
      "estimated_time_hours": 4
    }
  ],
  "total_steps": 4,
  "total_estimated_time_hours": 11.5,
  "path_summary": "Path from current competency (0.40 in Fractions) to target (Linear Equations) respecting prerequisites"
}
```

#### 5. Analyze Gaps

**Endpoint**: `POST /analyze/gaps`

**Request**:
```json
{
    "student_responses": [
    {
      "question_id": "Q001",
      "competency_score": 0.35,
      "question_text": "What is 1/2 + 1/3?",
      "subject": "Mathematics",
      "chapter": "Fractions",
      "learning_objective": "Add fractions with different denominators",
      "bloom_tag": "Apply",
      "response_time_seconds": 120,
      "attempts": 2
    },
    {
      "question_id": "Q002",
      "competency_score": 0.72,
      "question_text": "Convert 0.75 to a fraction",
      "subject": "Mathematics",
      "chapter": "Decimals",
      "learning_objective": "Convert between decimals and fractions",
      "bloom_tag": "Understand",
      "response_time_seconds": 45,
      "attempts": 1
    },
    {
      "question_id": "Q003",
      "competency_score": 0.28,
      "question_text": "Solve for x: x + 5 = 12",
      "subject": "Mathematics",
      "chapter": "Algebra",
      "learning_objective": "Solve one-step equations",
      "bloom_tag": "Apply",
      "response_time_seconds": 180,
      "attempts": 3
    }
    ],
    "competency_threshold": 0.6
}
```

**Response**:
```json
{
  "gaps": {
    "by_subject": {
      "Mathematics": {
        "avg_competency": 0.45,
        "total_questions": 3,
        "low_competency_count": 2,
        "gap_severity": 0.15,
        "rank": 0.15
      }
    },
    "by_chapter": {
      "Fractions": {
        "avg_competency": 0.35,
        "total_questions": 1,
        "low_competency_count": 1,
        "gap_severity": 0.25,
        "rank": 0.25
      },
      "Algebra": {
        "avg_competency": 0.28,
        "total_questions": 1,
        "low_competency_count": 1,
        "gap_severity": 0.32,
        "rank": 0.32
      }
    },
    "by_learning_objective": {
      "Add fractions with different denominators": {
        "avg_competency": 0.35,
        "total_questions": 1,
        "low_competency_count": 1,
        "gap_severity": 0.25,
        "rank": 0.25
      },
      "Solve one-step equations": {
        "avg_competency": 0.28,
        "total_questions": 1,
        "low_competency_count": 1,
        "gap_severity": 0.32,
        "rank": 0.32
      }
    },
    "by_bloom": {
      "Apply": {
        "avg_competency": 0.32,
        "total_questions": 2,
        "low_competency_count": 2,
        "gap_severity": 0.28,
        "rank": 0.28
      }
    },
    "root_causes": [
      {
        "type": "prerequisite_gap",
        "chapter": "Algebra",
        "prerequisite": "fractions_basic",
        "severity": 0.32,
        "description": "Weak foundation in fractions may be causing difficulty with algebra"
      },
      {
        "type": "prerequisite_gap",
        "chapter": "Fractions",
        "prerequisite": "basic_arithmetic",
        "severity": 0.25,
        "description": "Basic arithmetic skills need reinforcement"
      }
    ],
    "misconceptions": [
      {
        "misconception_id": "MC_FRAC_001",
        "type": "fraction_addition_error",
        "description": "Difficulty adding fractions with different denominators",
        "affected_chapters": ["Fractions"],
        "severity": "high"
      }
    ]
  },
  "root_causes": [
    {
      "type": "prerequisite_gap",
      "chapter": "Algebra",
      "prerequisite": "fractions_basic",
      "severity": 0.32,
      "description": "Weak foundation in fractions may be causing difficulty with algebra"
    }
  ],
  "remediation_strategy": {
    "priority_areas": [
      {
        "area": "Fractions",
        "priority": "high",
        "reason": "Highest gap severity and prerequisite for algebra",
        "recommended_actions": [
          "Review basic fraction concepts",
          "Practice adding fractions with same denominator first",
          "Gradually introduce different denominators"
        ],
        "estimated_remediation_time_hours": 4
      },
      {
        "area": "Algebra",
        "priority": "high",
        "reason": "Low competency and depends on fractions",
        "recommended_actions": [
          "Strengthen fraction skills first",
          "Introduce variables gradually",
          "Practice simple one-step equations"
        ],
        "estimated_remediation_time_hours": 6
      }
    ],
    "learning_path": [
      "basic_arithmetic",
      "fractions_basic",
      "fractions_advanced",
      "algebra_intro",
      "linear_equations"
    ],
    "recommended_questions_count": 25
  }
}
```

#### 6. Detect Misconceptions

**Endpoint**: `POST /detect/misconceptions`

**Request**:
```json
{
  "student_responses": [
    {
      "question_id": "Q001",
      "competency_score": 0.25,
      "question_text": "What is 1/2 + 1/3? Show your work.",
      "student_answer": "2/5",
      "response_time_seconds": 90,
      "attempts": 1
    },
    {
      "question_id": "Q002",
      "competency_score": 0.15,
      "question_text": "Identify the mistake: 3/4 + 1/2 = 4/6",
      "student_answer": "No mistake",
      "response_time_seconds": 120,
      "attempts": 2
    }
  ]
}
```

**Response**:
```json
{
  "misconceptions": [
    {
      "misconception_id": "MC_FRAC_ADD_001",
      "type": "fraction_addition_numerator_denominator",
      "description": "Student adds numerators and denominators separately when adding fractions",
      "pattern": "Adding fractions incorrectly: a/b + c/d = (a+c)/(b+d)",
      "evidence": [
        {
          "question_id": "Q001",
          "student_answer": "2/5",
          "correct_answer": "5/6",
          "confidence": 0.95
        },
        {
          "question_id": "Q002",
          "student_answer": "No mistake",
          "correct_answer": "Should be 5/4",
          "confidence": 0.88
        }
      ],
      "affected_concepts": ["fraction_addition", "fraction_operations"],
      "severity": "high",
      "frequency": 2,
      "remediation_resources": [
        {
          "type": "explanation",
          "content": "When adding fractions, you must first find a common denominator"
        },
        {
          "type": "practice_questions",
          "count": 10,
          "focus": "fraction_addition_with_common_denominator"
        }
      ]
    }
  ],
  "total_identified": 1,
  "remediation": {
    "priority_misconceptions": [
      {
        "misconception_id": "MC_FRAC_ADD_001",
        "priority": "high",
        "impact": "Blocks understanding of all fraction operations",
        "recommended_approach": "Direct instruction on common denominators followed by guided practice"
      }
    ],
    "recommended_questions": [
      {
        "question_id": "Q_R001",
        "reason": "Addresses misconception MC_FRAC_ADD_001",
        "focus": "Finding common denominators"
      }
    ],
    "estimated_remediation_time_hours": 3
  }
}
```

#### 7. Query Knowledge Graph

**Get Concept Information**

**Endpoint**: `GET /graph/concepts/{concept_id}`

**Example**: `GET /graph/concepts/chapter_Fractions`

**Response**:
```json
{
  "concept_details": {
    "concept_id": "chapter_Fractions",
    "name": "Fractions",
    "type": "chapter",
    "subject": "Mathematics",
    "grade_levels": ["5", "6", "7"],
    "description": "Understanding and operations with fractions",
    "prerequisites": [
      "chapter_Basic_Arithmetic",
      "concept_division"
    ],
    "prerequisite_of": [
      "chapter_Decimals",
      "chapter_Algebra"
    ],
    "learning_objectives": [
      "Understand fraction representation",
      "Add and subtract fractions",
      "Multiply and divide fractions"
    ],
    "difficulty_level": 0.65,
    "estimated_time_hours": 8,
    "misconceptions": [
      {
        "misconception_id": "MC_FRAC_ADD_001",
        "description": "Adding numerators and denominators separately",
        "frequency": 0.35
      }
    ],
    "related_concepts": [
      "chapter_Decimals",
      "chapter_Percentages"
    ],
    "node_centrality": 0.82,
    "embedding": [0.12, -0.34, 0.56, ...]
  }
}
```

**Get Prerequisites**

**Endpoint**: `GET /graph/prerequisites/{concept_id}?max_depth=3`

**Example**: `GET /graph/prerequisites/chapter_Algebra?max_depth=3`

**Response**:
```json
{
  "prerequisites": [
    "chapter_Fractions",
    "chapter_Decimals",
    "chapter_Basic_Arithmetic",
    "concept_division",
    "concept_multiplication"
  ]
}
```

**Find Learning Path Between Concepts**

**Endpoint**: `GET /graph/path/{from_concept}/{to_concept}?respect_prerequisites=true`

**Example**: `GET /graph/path/chapter_Fractions/chapter_Algebra?respect_prerequisites=true`

**Response**:
```json
{
  "path": [
    "chapter_Fractions",
    "chapter_Decimals",
    "chapter_Algebra_Basics",
    "chapter_Algebra"
  ],
  "path_length": 4,
  "estimated_time_hours": 12,
  "prerequisites_met": true
}
```

**Get Misconceptions for Concept**

**Endpoint**: `GET /graph/misconceptions/{concept_id}`

**Example**: `GET /graph/misconceptions/chapter_Fractions`

**Response**:
```json
{
  "misconceptions": [
    {
      "misconception_id": "MC_FRAC_ADD_001",
      "description": "Adding numerators and denominators separately",
      "pattern": "a/b + c/d = (a+c)/(b+d)",
      "frequency": 0.35,
      "severity": "high"
    },
    {
      "misconception_id": "MC_FRAC_MULT_001",
      "description": "Multiplying denominators when adding fractions",
      "pattern": "a/b + c/d = (a+c)/(b*d)",
      "frequency": 0.22,
      "severity": "medium"
    }
  ]
}
```

## Configuration

Edit `config.yaml` to customize:

- **Data paths**: Input file, train/val/test splits
- **Model parameters**: Algorithm choices, hyperparameters
- **NLP settings**: Model names, batch sizes
- **Knowledge graph**: Embedding dimensions, walk parameters
- **API settings**: Host, port, workers

## Knowledge Graph Visualization

```python
from src.knowledge_graph.visualization import KnowledgeGraphVisualizer

visualizer = KnowledgeGraphVisualizer(knowledge_graph)

# Create interactive visualization
visualizer.visualize_interactive(output_path='knowledge_graph.html')

# Visualize subgraph
visualizer.visualize_subgraph(
    nodes=['grade_6', 'subject_Mathematics', 'chapter_Fractions'],
    output_path='subgraph.png'
)
```

## Model Cards

Each model includes performance metrics and limitations. See `models/` directory for exported model cards after training.

## Examples

See `notebooks/` directory for Jupyter notebooks with detailed examples:
- Data exploration and analysis
- Knowledge graph construction
- Model training and evaluation
- API usage examples

## Testing

```bash
pytest tests/
```


```
Behavioral Model for Adaptive Learning System
A comprehensive system for predicting competency, detecting misconceptions, 
and generating adaptive learning paths using knowledge graphs.
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
