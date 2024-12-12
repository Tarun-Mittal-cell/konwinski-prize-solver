from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import traceback
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Issue:
    """GitHub issue representation"""
    id: str
    title: str
    body: str
    files: List[str]
    tests: List[str]
    repo: str
    base_commit: str

@dataclass
class Solution:
    """Proposed fix for an issue"""
    files_modified: Dict[str, str]  # filename -> new content
    confidence: float
    validation_status: bool = False
    error_message: Optional[str] = None

class GithubIssueAgent:
    """Production agent for fixing GitHub issues"""
    
    def __init__(
        self, 
        model_name: str = "codellama/CodeLlama-13b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        
        # Configuration
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.95
        self.min_confidence = 0.8
        
        # State tracking
        self.total_processed = 0
        self.successful_fixes = 0
        self.cached_data = {}
        
    def setup(self) -> None:
        """Initialize models and resources"""
        try:
            logger.info(f"Loading model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Model initialization failed")
            
    def load_dataset(self, split: str = 'test') -> List[Issue]:
        """Load SWE-bench dataset"""
        try:
            logger.info(f"Loading SWE-bench {split} split")
            data = load_dataset('princeton-nlp/SWE-bench', split=split)
            
            issues = []
            for item in data:
                issue = Issue(
                    id=item['id'],
                    title=item['title'],
                    body=item['body'],
                    files=item['files'],
                    tests=item['tests'],
                    repo=item['repo'],
                    base_commit=item['base_commit']
                )
                issues.append(issue)
                
            logger.info(f"Loaded {len(issues)} issues")
            return issues
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Dataset loading failed")

    def should_attempt_fix(self, issue: Issue) -> bool:
        """Determine if we should attempt to fix this issue"""
        try:
            # Skip if too many files need changes
            if len(issue.files) > 3:
                logger.debug(f"Skipping {issue.id}: Too many files ({len(issue.files)})")
                return False
                
            # Skip if non-Python files
            if not all(f.endswith('.py') for f in issue.files):
                logger.debug(f"Skipping {issue.id}: Non-Python files")
                return False
                
            # Skip if test files are missing
            if not issue.tests:
                logger.debug(f"Skipping {issue.id}: No test files")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in should_attempt_fix: {str(e)}")
            return False

    def generate_fix(self, issue: Issue) -> Optional[Solution]:
        """Generate a fix for the given issue"""
        try:
            if not self.should_attempt_fix(issue):
                return None
                
            # Create prompt
            prompt = self._create_prompt(issue)
            
            # Generate solution
            response = self._generate_solution(prompt)
            if not response:
                return None
                
            # Parse and validate solution
            solution = self._parse_solution(response, issue)
            if not solution or solution.confidence < self.min_confidence:
                return None
                
            # Validate the solution
            solution.validation_status = self._validate_solution(solution, issue)
            if not solution.validation_status:
                return None
                
            self.successful_fixes += 1
            return solution
            
        except Exception as e:
            logger.error(f"Error generating fix for {issue.id}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _create_prompt(self, issue: Issue) -> str:
        """Create detailed prompt for the model"""
        return f"""Fix this GitHub issue:
Repository: {issue.repo}
Issue Title: {issue.title}

Description:
{issue.body}

Files to modify: {', '.join(issue.files)}
Tests: {', '.join(issue.tests)}

Generate a solution that includes only the necessary changes to fix the issue.
Format the response as:
<SOLUTION>
[file path]
[code changes]
</SOLUTION>"""

    def _generate_solution(self, prompt: str) -> Optional[str]:
        """Generate solution using the model"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error in solution generation: {str(e)}")
            return None

    def _parse_solution(self, response: str, issue: Issue) -> Optional[Solution]:
        """Parse model response into structured solution"""
        try:
            # Extract solution content
            if "<SOLUTION>" not in response or "</SOLUTION>" not in response:
                return None
                
            content = response.split("<SOLUTION>")[1].split("</SOLUTION>")[0].strip()
            
            # Parse modified files
            files_modified = {}
            current_file = None
            current_content = []
            
            for line in content.split("\n"):
                if line.strip() and line.endswith(".py"):
                    if current_file and current_content:
                        files_modified[current_file] = "\n".join(current_content)
                    current_file = line.strip()
                    current_content = []
                elif current_file:
                    current_content.append(line)
                    
            if current_file and current_content:
                files_modified[current_file] = "\n".join(current_content)
                
            # Basic confidence scoring
            confidence = self._calculate_confidence(files_modified, issue)
            
            return Solution(
                files_modified=files_modified,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing solution: {str(e)}")
            return None

    def _calculate_confidence(self, files_modified: Dict[str, str], issue: Issue) -> float:
        """Calculate confidence score for the solution"""
        try:
            score = 1.0
            
            # Reduce confidence if we modify files not mentioned
            for file in files_modified:
                if file not in issue.files:
                    score *= 0.5
                    
            # Reduce confidence for very large changes
            for content in files_modified.values():
                if len(content.split("\n")) > 50:
                    score *= 0.7
                    
            return score
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _validate_solution(self, solution: Solution, issue: Issue) -> bool:
        """Validate the solution (placeholder - implement actual testing)"""
        try:
            # Implement actual test running logic here
            return True
            
        except Exception as e:
            logger.error(f"Error in solution validation: {str(e)}")
            solution.error_message = str(e)
            return False
            
    def save_statistics(self, path: str) -> None:
        """Save agent statistics to file"""
        try:
            stats = {
                "total_processed": self.total_processed,
                "successful_fixes": self.successful_fixes,
                "success_rate": self.successful_fixes / max(1, self.total_processed)
            }
            
            with open(path, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving statistics: {str(e)}")

def main():
    """Example usage"""
    try:
        # Initialize agent
        agent = GithubIssueAgent()
        agent.setup()
        
        # Load dataset
        issues = agent.load_dataset('test')
        
        # Process issues
        for issue in issues:
            solution = agent.generate_fix(issue)
            if solution and solution.validation_status:
                logger.info(f"Successfully fixed issue {issue.id}")
        
        # Save statistics
        agent.save_statistics("agent_stats.json")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

