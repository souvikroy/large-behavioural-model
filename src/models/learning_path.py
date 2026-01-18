"""Learning path generator."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import networkx as nx

from .competency_predictor import CompetencyPredictor
from ..knowledge_graph.query import KnowledgeGraphQuery


class LearningPathGenerator:
    """Generate optimal learning paths using knowledge graph."""
    
    def __init__(
        self,
        competency_predictor: CompetencyPredictor,
        knowledge_graph_query: KnowledgeGraphQuery,
        config_path: str = "config.yaml"
    ):
        """
        Initialize learning path generator.
        
        Args:
            competency_predictor: CompetencyPredictor instance
            knowledge_graph_query: KnowledgeGraphQuery instance
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.path_config = self.config.get('models', {}).get('learning_path', {})
        self.max_path_length = self.path_config.get('max_path_length', 20)
        self.respect_prerequisites = self.path_config.get('respect_prerequisites', True)
        self.avoid_misconceptions = self.path_config.get('avoid_misconceptions', True)
        
        self.competency_predictor = competency_predictor
        self.graph_query = knowledge_graph_query
    
    def generate_path(
        self,
        student_profile: Dict[str, any],
        target_concept: Optional[str] = None,
        target_chapter: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Generate learning path for student.
        
        Args:
            student_profile: Student competency profile
            target_concept: Optional target concept
            target_chapter: Optional target chapter
            
        Returns:
            List of learning steps in the path
        """
        path = []
        
        # Get current competency by dimension
        competency_by_subject = student_profile.get('competency_by_subject', {})
        competency_by_chapter = student_profile.get('competency_by_chapter', {})
        mastered_concepts = student_profile.get('mastered_concepts', [])
        
        if target_chapter:
            # Generate path to specific chapter
            path = self._generate_chapter_path(
                target_chapter,
                competency_by_chapter,
                mastered_concepts
            )
        elif target_concept:
            # Generate path to specific concept
            path = self._generate_concept_path(
                target_concept,
                mastered_concepts
            )
        else:
            # Generate general learning path
            path = self._generate_general_path(
                competency_by_subject,
                competency_by_chapter,
                mastered_concepts
            )
        
        return path
    
    def _generate_chapter_path(
        self,
        target_chapter: str,
        competency_by_chapter: Dict[str, float],
        mastered_concepts: List[str]
    ) -> List[Dict[str, any]]:
        """
        Generate path to a specific chapter.
        
        Args:
            target_chapter: Target chapter name
            competency_by_chapter: Competency scores by chapter
            mastered_concepts: List of mastered concept IDs
            
        Returns:
            List of path steps
        """
        path = []
        
        # Find chapter node
        chapter_node = f"chapter_{target_chapter}"
        
        if chapter_node not in self.graph_query.graph:
            return path
        
        # Get prerequisites
        prerequisites = self.graph_query.find_prerequisites(chapter_node)
        
        # Build path respecting prerequisites
        for prereq in prerequisites:
            prereq_data = self.graph_query.graph.nodes[prereq]
            if prereq_data.get('node_type') == 'concept':
                path.append({
                    'step': len(path) + 1,
                    'type': 'concept',
                    'concept': prereq_data.get('name', prereq),
                    'reason': 'Prerequisite for target chapter'
                })
        
        # Add target chapter
        path.append({
            'step': len(path) + 1,
            'type': 'chapter',
            'chapter': target_chapter,
            'reason': 'Target chapter'
        })
        
        return path
    
    def _generate_concept_path(
        self,
        target_concept: str,
        mastered_concepts: List[str]
    ) -> List[Dict[str, any]]:
        """
        Generate path to a specific concept.
        
        Args:
            target_concept: Target concept name
            mastered_concepts: List of mastered concept IDs
            
        Returns:
            List of path steps
        """
        path = []
        
        # Find concept node
        concept_node = f"concept_{hash(target_concept)}"
        
        if concept_node not in self.graph_query.graph:
            return path
        
        # Get prerequisites
        prerequisites = self.graph_query.find_prerequisites(concept_node)
        
        # Filter out already mastered
        remaining_prereqs = [p for p in prerequisites if p not in mastered_concepts]
        
        # Build path
        for prereq in remaining_prereqs:
            prereq_data = self.graph_query.graph.nodes[prereq]
            path.append({
                'step': len(path) + 1,
                'type': 'concept',
                'concept': prereq_data.get('name', prereq),
                'reason': 'Prerequisite for target concept'
            })
        
        # Add target concept
        path.append({
            'step': len(path) + 1,
            'type': 'concept',
            'concept': target_concept,
            'reason': 'Target concept'
        })
        
        return path
    
    def _generate_general_path(
        self,
        competency_by_subject: Dict[str, float],
        competency_by_chapter: Dict[str, float],
        mastered_concepts: List[str]
    ) -> List[Dict[str, any]]:
        """
        Generate general learning path based on competency gaps.
        
        Args:
            competency_by_subject: Competency by subject
            competency_by_chapter: Competency by chapter
            mastered_concepts: List of mastered concepts
            
        Returns:
            List of path steps
        """
        path = []
        
        # Identify weak areas
        weak_subjects = [
            subject for subject, comp in competency_by_subject.items()
            if comp < 0.6
        ]
        
        weak_chapters = [
            chapter for chapter, comp in competency_by_chapter.items()
            if comp < 0.6
        ]
        
        # Build path focusing on weak areas
        for subject in weak_subjects[:3]:  # Limit to top 3
            path.append({
                'step': len(path) + 1,
                'type': 'subject',
                'subject': subject,
                'reason': f'Weak area (competency: {competency_by_subject[subject]:.2f})'
            })
            
            # Add weak chapters in this subject
            subject_chapters = [
                ch for ch in weak_chapters
                if ch.startswith(subject) or True  # Simplified
            ]
            
            for chapter in subject_chapters[:2]:  # Limit chapters
                path.append({
                    'step': len(path) + 1,
                    'type': 'chapter',
                    'chapter': chapter,
                    'reason': f'Weak chapter in {subject}'
                })
        
        return path[:self.max_path_length]
