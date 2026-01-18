# Streamlit Frontend for Question Quest

A modern, interactive web interface for the Behavioral Model for Adaptive Learning System.

## Features

- 🎯 **Competency Prediction**: Predict student competency scores for questions
- ❓ **Question Recommendations**: Get personalized question recommendations based on student profiles
- 🛤️ **Learning Path Generation**: Generate optimal learning paths for students
- 📊 **Gap Analysis**: Analyze competency gaps and identify root causes
- 🔍 **Misconception Detection**: Detect misconceptions from student responses
- 🌐 **Knowledge Graph Explorer**: Explore educational concepts and their relationships

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install from the main project root:

```bash
pip install streamlit requests pandas plotly
```

## Running the Frontend

1. Ensure the FastAPI backend is running (default: http://localhost:9010)

2. Start the Streamlit app:

```bash
streamlit run app.py
```

Or from the project root:

```bash
streamlit run frontend/app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. **Configure API URL**: Use the sidebar to set the API base URL if your backend is running on a different host/port

2. **Navigate**: Use the sidebar navigation to access different features:
   - **Home**: Overview and API status check
   - **Competency Prediction**: Enter question details to predict competency
   - **Question Recommendations**: Provide student profile to get recommendations
   - **Learning Path Generation**: Generate personalized learning paths
   - **Gap Analysis**: Analyze student responses to identify gaps
   - **Misconception Detection**: Detect misconceptions from responses
   - **Knowledge Graph Explorer**: Explore concepts, prerequisites, and paths

3. **View Results**: Results are displayed with interactive visualizations and detailed information

## API Endpoints Used

The frontend interacts with the following FastAPI endpoints:

- `GET /health` - Health check
- `POST /predict/competency` - Competency prediction
- `POST /recommend/questions` - Question recommendations
- `POST /generate/path` - Learning path generation
- `POST /analyze/gaps` - Gap analysis
- `POST /detect/misconceptions` - Misconception detection
- `GET /graph/concepts/{concept_id}` - Get concept details
- `GET /graph/prerequisites/{concept_id}` - Get prerequisites
- `GET /graph/path/{from_concept}/{to_concept}` - Find learning path

## Configuration

- Default API URL: `http://localhost:9010`
- Can be changed via the sidebar input field
- All API requests timeout after 30 seconds

## Troubleshooting

- **Cannot connect to API**: Ensure the FastAPI backend is running and accessible at the configured URL
- **Empty results**: Check that models are loaded in the backend (see backend logs)
- **Timeout errors**: Increase timeout values or check network connectivity

## Development

To modify the frontend:

1. Edit `app.py` to add new features or modify existing ones
2. Use `streamlit run app.py` to see changes in real-time
3. Check Streamlit documentation for available components and features
