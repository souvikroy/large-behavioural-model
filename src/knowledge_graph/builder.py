"""Knowledge graph builder."""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import yaml
from collections import defaultdict

from ..models.misconception_detector import MisconceptionDetector
from ..nlp.concept_extractor import ConceptExtractor


class KnowledgeGraphBuilder:
    """Build knowledge graph from educational data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize knowledge graph builder.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kg_config = self.config.get('knowledge_graph', {})
        self.graph = nx.MultiDiGraph()
        self.concept_extractor = ConceptExtractor()
        self.misconception_detector = None
        
        # Node type mappings
        self.node_types = {
            'grade', 'subject', 'chapter', 'learning_objective', 
            'learning_unit', 'question', 'concept', 'misconception', 'bloom_level'
        }
    
    def build_hierarchical_structure(self, df: pd.DataFrame) -> None:
        """
        Build hierarchical structure: Grade → Subject → Chapter → LO → LU.
        
        Args:
            df: DataFrame with educational data
        """
        # Create nodes and edges for hierarchy
        for _, row in df.iterrows():
            grade = str(row.get('grade', ''))
            subject = str(row.get('Subject', ''))
            chapter = str(row.get('chapter', ''))
            lo = str(row.get('learning_objective', ''))
            lu = str(row.get('learning_unit', ''))
            question_id = str(row.get('question_id', ''))
            bloom = str(row.get('Bloom_tag', ''))
            
            # Create nodes
            if grade:
                self.graph.add_node(f"grade_{grade}", node_type='grade', name=grade)
            if subject:
                self.graph.add_node(f"subject_{subject}", node_type='subject', name=subject)
            if chapter:
                self.graph.add_node(f"chapter_{chapter}", node_type='chapter', name=chapter)
            if lo:
                self.graph.add_node(f"lo_{hash(lo)}", node_type='learning_objective', name=lo)
            if lu:
                self.graph.add_node(f"lu_{hash(lu)}", node_type='learning_unit', name=lu)
            if question_id:
                competency = pd.to_numeric(row.get('competency_score', 0), errors='coerce')
                self.graph.add_node(
                    f"question_{question_id}",
                    node_type='question',
                    question_id=question_id,
                    question_text=str(row.get('question_text', '')),
                    competency_score=competency if not pd.isna(competency) else None,
                    bloom_tag=bloom,
                    subject=subject,
                    chapter=chapter
                )
            if bloom:
                self.graph.add_node(f"bloom_{bloom}", node_type='bloom_level', name=bloom)
            
            # Create hierarchical edges
            if grade and subject:
                self.graph.add_edge(
                    f"grade_{grade}", 
                    f"subject_{subject}",
                    edge_type='CONTAINS',
                    weight=1.0
                )
            if subject and chapter:
                self.graph.add_edge(
                    f"subject_{subject}",
                    f"chapter_{chapter}",
                    edge_type='CONTAINS',
                    weight=1.0
                )
            if chapter and lo:
                self.graph.add_edge(
                    f"chapter_{chapter}",
                    f"lo_{hash(lo)}",
                    edge_type='CONTAINS',
                    weight=1.0
                )
            if lo and lu:
                self.graph.add_edge(
                    f"lo_{hash(lo)}",
                    f"lu_{hash(lu)}",
                    edge_type='CONTAINS',
                    weight=1.0
                )
            if lu and question_id:
                self.graph.add_edge(
                    f"lu_{hash(lu)}",
                    f"question_{question_id}",
                    edge_type='HAS_QUESTION',
                    weight=1.0
                )
            if question_id and bloom:
                self.graph.add_edge(
                    f"question_{question_id}",
                    f"bloom_{bloom}",
                    edge_type='TESTS',
                    weight=1.0
                )
    
    def extract_and_link_concepts(self, df: pd.DataFrame) -> None:
        """
        Extract concepts from question text and link them.
        
        Args:
            df: DataFrame with questions
        """
        # Batch collect question data for optimization
        question_data = []
        for _, row in df.iterrows():
            question_id = str(row.get('question_id', ''))
            question_text = str(row.get('question_text', ''))
            
            if question_id and question_text:
                question_data.append((question_id, question_text))
        
        # Process in batch (concept extractor will handle batching internally)
        # Extract concepts for all questions
        question_texts = [q_text for _, q_text in question_data]
        all_concepts = self.concept_extractor.extract_concepts_batch(question_texts)
        
        # Link concepts to questions (same logic as before, just batched)
        for (question_id, question_text), concepts in zip(question_data, all_concepts):
            for concept in concepts:
                concept_node = f"concept_{hash(concept)}"
                
                # Create concept node
                if concept_node not in self.graph:
                    self.graph.add_node(concept_node, node_type='concept', name=concept)
                
                # Link question to concept
                if f"question_{question_id}" in self.graph:
                    self.graph.add_edge(
                        f"question_{question_id}",
                        concept_node,
                        edge_type='TESTS',
                        weight=1.0
                    )
    
    def detect_prerequisites(self, df: pd.DataFrame) -> None:
        """
        Detect prerequisite relationships between concepts.
        
        Args:
            df: DataFrame with questions and competency scores
        """
        df = df.copy()
        df['competency_score'] = pd.to_numeric(df['competency_score'], errors='coerce')
        
        # Method 1: Competency-based inference
        # Group by concept and calculate average competency
        concept_competency = defaultdict(list)
        
        # Batch extract concepts for all questions
        question_texts = df['question_text'].astype(str).tolist()
        all_concepts_list = self.concept_extractor.extract_concepts_batch(question_texts)
        
        # Process results
        for idx, (_, row) in enumerate(df.iterrows()):
            competency = row.get('competency_score')
            
            if pd.isna(competency):
                continue
            
            concepts = all_concepts_list[idx] if idx < len(all_concepts_list) else set()
            for concept in concepts:
                concept_competency[concept].append(competency)
        
        # Calculate average competency per concept
        concept_avg_competency = {
            concept: np.mean(scores)
            for concept, scores in concept_competency.items()
        }
        
        # Method 2: Sequential analysis (grade progression)
        grade_concepts = defaultdict(set)
        # Use already extracted concepts from above
        for idx, (_, row) in enumerate(df.iterrows()):
            grade = str(row.get('grade', ''))
            concepts = all_concepts_list[idx] if idx < len(all_concepts_list) else set()
            grade_concepts[grade].update(concepts)
        
        # Create prerequisite edges based on grade progression
        grades = sorted([g for g in grade_concepts.keys() if g.isdigit()])
        for i in range(len(grades) - 1):
            prev_grade = grades[i]
            next_grade = grades[i + 1]
            
            prev_concepts = grade_concepts[prev_grade]
            next_concepts = grade_concepts[next_grade]
            
            # Concepts from previous grade might be prerequisites
            for prev_concept in prev_concepts:
                for next_concept in next_concepts:
                    if prev_concept != next_concept:
                        prev_node = f"concept_{hash(prev_concept)}"
                        next_node = f"concept_{hash(next_concept)}"
                        
                        if prev_node in self.graph and next_node in self.graph:
                            # Check if competency suggests prerequisite
                            prev_comp = concept_avg_competency.get(prev_concept, 0.5)
                            next_comp = concept_avg_competency.get(next_concept, 0.5)
                            
                            # If previous concept has higher competency, it might be prerequisite
                            if prev_comp > next_comp + 0.1:  # Threshold
                                self.graph.add_edge(
                                    prev_node,
                                    next_node,
                                    edge_type='PREREQUISITE',
                                    weight=prev_comp - next_comp,
                                    confidence=min(prev_comp - next_comp, 1.0)
                                )
        
        # Method 3: Bloom taxonomy progression
        bloom_order = {'Remember': 1, 'Understand': 2, 'Apply': 3, 'Analyse': 4, 'Evaluate': 5}
        
        bloom_concepts = defaultdict(set)
        # Use already extracted concepts from above
        for idx, (_, row) in enumerate(df.iterrows()):
            bloom = str(row.get('Bloom_tag', ''))
            concepts = all_concepts_list[idx] if idx < len(all_concepts_list) else set()
            if bloom in bloom_order:
                bloom_concepts[bloom].update(concepts)
        
        # Create prerequisite edges based on Bloom progression
        bloom_levels = sorted(bloom_concepts.keys(), key=lambda x: bloom_order.get(x, 0))
        for i in range(len(bloom_levels) - 1):
            prev_bloom = bloom_levels[i]
            next_bloom = bloom_levels[i + 1]
            
            prev_concepts = bloom_concepts[prev_bloom]
            next_concepts = bloom_concepts[next_bloom]
            
            for prev_concept in prev_concepts:
                for next_concept in next_concepts:
                    if prev_concept == next_concept:  # Same concept at different levels
                        prev_node = f"concept_{hash(prev_concept)}"
                        next_node = f"concept_{hash(next_concept)}"
                        
                        if prev_node in self.graph and next_node in self.graph:
                            self.graph.add_edge(
                                prev_node,
                                next_node,
                                edge_type='PROGRESSES_TO',
                                weight=1.0
                            )
    
    def integrate_misconceptions(self, misconception_detector: MisconceptionDetector) -> None:
        """
        Integrate misconceptions into knowledge graph.
        
        Args:
            misconception_detector: MisconceptionDetector instance
        """
        self.misconception_detector = misconception_detector
        
        misconceptions = misconception_detector.misconceptions
        
        for mc_id, mc_data in misconceptions.items():
            concept = mc_data.get('concept', '')
            misconception_node = f"misconception_{mc_id}"
            
            # Create misconception node
            self.graph.add_node(
                misconception_node,
                node_type='misconception',
                misconception_id=mc_id,
                concept=concept,
                misconception_type=mc_data.get('type', 'Conceptual'),
                frequency=mc_data.get('frequency', 0),
                severity=mc_data.get('severity', 0.0)
            )
            
            # Link misconception to concept
            concept_node = f"concept_{hash(concept)}"
            if concept_node in self.graph:
                self.graph.add_edge(
                    misconception_node,
                    concept_node,
                    edge_type='ASSOCIATED_WITH',
                    weight=mc_data.get('severity', 0.0)
                )
            
            # Link misconception to questions
            for q_idx in mc_data.get('associated_questions', []):
                # Note: q_idx is DataFrame index, need to map to question_id
                # This is simplified - in practice, you'd maintain this mapping
                pass
    
    def add_difficulty_edges(self, df: pd.DataFrame) -> None:
        """
        Add difficulty progression edges based on 3PL scores.
        
        Args:
            df: DataFrame with questions and 3PL scores
        """
        df = df.copy()
        df['3PL'] = pd.to_numeric(df['3PL'], errors='coerce')
        
        # Get questions with 3PL scores
        questions_with_3pl = df[df['3PL'].notna()]
        
        # Sort by 3PL (lower = easier)
        questions_sorted = questions_with_3pl.sort_values('3PL')
        
        # Create easier_than edges between consecutive questions
        for i in range(len(questions_sorted) - 1):
            q1_id = str(questions_sorted.iloc[i]['question_id'])
            q2_id = str(questions_sorted.iloc[i + 1]['question_id'])
            
            q1_node = f"question_{q1_id}"
            q2_node = f"question_{q2_id}"
            
            if q1_node in self.graph and q2_node in self.graph:
                difficulty_diff = questions_sorted.iloc[i + 1]['3PL'] - questions_sorted.iloc[i]['3PL']
                if difficulty_diff > 0:  # q2 is harder
                    self.graph.add_edge(
                        q1_node,
                        q2_node,
                        edge_type='EASIER_THAN',
                        weight=difficulty_diff
                    )
    
    def build(self, df: pd.DataFrame, misconception_detector: Optional[MisconceptionDetector] = None) -> nx.MultiDiGraph:
        """
        Build complete knowledge graph.
        
        Args:
            df: DataFrame with educational data
            misconception_detector: Optional MisconceptionDetector instance
            
        Returns:
            Built knowledge graph
        """
        # Step 1: Build hierarchical structure
        self.build_hierarchical_structure(df)
        
        # Step 2: Extract and link concepts
        self.extract_and_link_concepts(df)
        
        # Step 3: Detect prerequisites
        self.detect_prerequisites(df)
        
        # Step 4: Integrate misconceptions
        if misconception_detector:
            self.integrate_misconceptions(misconception_detector)
        
        # Step 5: Add difficulty edges
        self.add_difficulty_edges(df)
        
        return self.graph
    
    def get_graph(self) -> nx.MultiDiGraph:
        """Get the built knowledge graph."""
        return self.graph
