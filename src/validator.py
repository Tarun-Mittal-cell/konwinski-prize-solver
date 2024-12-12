import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from swebench.harness.run_evaluation import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    instance_id: str
    status: str  # 'success', 'fail', 'skip'
    patch_content: Optional[str] = None
    test_outputs: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None
    reasoning_trace: Optional[str] = None

class SWEBenchValidator:
    def __init__(
        self,
        submission_dir: str = "submission",
        workspace_dir: str = "data/workspace"
    ):
        self.submission_dir = Path(submission_dir)
        self.workspace_dir = Path(workspace_dir)
        self.results: List[ValidationResult] = []
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary directories"""
        dirs = [
            self.submission_dir,
            self.submission_dir / "predictions",
            self.submission_dir / "trajs",
            self.submission_dir / "logs",
            self.workspace_dir
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("Created directories")

    def validate_solution(
        self,
        instance_id: str,
        patch_content: Optional[str],
        test_files: List[str],
        reasoning: str
    ) -> ValidationResult:
        """Validate a solution using SWE-bench"""
        try:
            # Handle skip case
            if not patch_content:
                return ValidationResult(
                    instance_id=instance_id,
                    status='skip',
                    reasoning_trace=reasoning
                )

            # Save files
            self._save_reasoning_trace(instance_id, reasoning)
            validation_result = self._run_swebench_validation(
                instance_id, patch_content, test_files
            )

            self.results.append(validation_result)
            return validation_result

        except Exception as e:
            logger.error(f"Validation failed for {instance_id}: {str(e)}")
            return ValidationResult(
                instance_id=instance_id,
                status='fail',
                error_message=str(e),
                reasoning_trace=reasoning
            )

    def _run_swebench_validation(
        self,
        instance_id: str,
        patch_content: str,
        test_files: List[str]
    ) -> ValidationResult:
        """Run validation using SWE-bench"""
        try:
            # Setup paths
            instance_dir = self.workspace_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            
            patch_path = instance_dir / "patch.diff"
            with open(patch_path, "w") as f:
                f.write(patch_content)

            # Run SWE-bench evaluation
            result = run_evaluation(
                instance_ids=[instance_id],
                patch_path=str(patch_path),
                max_workers=1
            )

            # Parse results
            status = 'success' if result.get('passed', False) else 'fail'
            test_outputs = result.get('test_outputs', {})
            error = result.get('error', None)

            return ValidationResult(
                instance_id=instance_id,
                status=status,
                patch_content=patch_content,
                test_outputs=test_outputs,
                error_message=error
            )

        except Exception as e:
            logger.error(f"SWE-bench validation failed: {str(e)}")
            return ValidationResult(
                instance_id=instance_id,
                status='fail',
                error_message=str(e)
            )

    def _save_reasoning_trace(self, instance_id: str, reasoning: str) -> None:
        """Save reasoning trace"""
        try:
            trace_path = self.submission_dir / "trajs" / f"{instance_id}.md"
            with open(trace_path, "w") as f:
                f.write(reasoning)
        except Exception as e:
            logger.error(f"Failed to save reasoning: {str(e)}")

    def calculate_score(self) -> float:
        """Calculate competition score"""
        try:
            a = sum(1 for r in self.results if r.status == 'success')
            b = sum(1 for r in self.results if r.status == 'fail')
            c = sum(1 for r in self.results if r.status == 'skip')
            
            if a + b + c == 0:
                return 0.0
            
            return (a - b) / (a + b + c)
            
        except Exception as e:
            logger.error(f"Score calculation failed: {str(e)}")
            return 0.0

    def save_submission(self) -> None:
        """Save Kaggle submission files"""
        try:
            predictions = {
                r.instance_id: {
                    "status": r.status,
                    "patch": r.patch_content if r.patch_content else "",
                    "error": r.error_message if r.error_message else ""
                }
                for r in self.results
            }
            
            with open(self.submission_dir / "predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

            score = self.calculate_score()
            with open(self.submission_dir / "score.json", "w") as f:
                json.dump({"score": score}, f, indent=2)

            logger.info(f"Saved submission (score: {score:.4f})")

        except Exception as e:
            logger.error(f"Failed to save submission: {str(e)}")
            raise

def main():
    """Test validator"""
    validator = SWEBenchValidator()
    
    # Test validation
    result = validator.validate_solution(
        instance_id="test-1",
        patch_content="Example patch",
        test_files=["test1.py"],
        reasoning="Test reasoning"
    )
    
    validator.save_submission()
    print(f"Score: {validator.calculate_score():.4f}")

if __name__ == "__main__":
    main()

