import os
import json
import logging
import docker
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Stores validation results for an issue fix"""
    instance_id: str
    passed: bool
    test_output: str
    reasoning_trace: str
    error_message: Optional[str] = None

class SWEValidator:
    def __init__(self, workspace_dir: str = "data/workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.docker_client = docker.from_client()
        self.submission_dir = Path("submission")
        self.setup_directories()
        
    def setup_directories(self):
        """Create required directories"""
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        os.makedirs(self.submission_dir / "logs", exist_ok=True)
        os.makedirs(self.submission_dir / "trajs", exist_ok=True)

    def validate_solution(self, instance_id: str, patch_content: str, reasoning: str) -> ValidationResult:
        """Validate a solution using Docker"""
        try:
            # Create instance workspace
            instance_dir = self.workspace_dir / instance_id
            os.makedirs(instance_dir, exist_ok=True)
            
            # Save patch
            patch_path = instance_dir / "patch.diff"
            with open(patch_path, "w") as f:
                f.write(patch_content)
            
            # Save reasoning trace
            traj_path = self.submission_dir / "trajs" / f"{instance_id}.md"
            with open(traj_path, "w") as f:
                f.write(reasoning)
            
            # Run validation in Docker
            result = self._run_docker_validation(instance_id, patch_path)
            
            # Save logs
            self._save_logs(instance_id, result)
            
            return ValidationResult(
                instance_id=instance_id,
                passed=result["passed"],
                test_output=result["test_output"],
                reasoning_trace=reasoning,
                error_message=result.get("error")
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {instance_id}: {str(e)}")
            return ValidationResult(
                instance_id=instance_id,
                passed=False,
                test_output="",
                reasoning_trace=reasoning,
                error_message=str(e)
            )

    def _run_docker_validation(self, instance_id: str, patch_path: Path) -> Dict:
        """Run validation inside Docker container"""
        try:
            # Pull SWE-bench evaluation image
            self.docker_client.images.pull("swebench/evaluation:latest")
            
            # Run validation
            container = self.docker_client.containers.run(
                "swebench/evaluation:latest",
                command=f"python -m swebench.harness.run_evaluation "
                       f"--instance_ids {instance_id} "
                       f"--patch_path {patch_path}",
                volumes={
                    str(self.workspace_dir): {"bind": "/workspace", "mode": "rw"}
                },
                detach=True
            )
            
            # Wait for completion
            container.wait()
            
            # Get logs
            logs = container.logs().decode()
            
            # Parse results
            results_path = self.workspace_dir / instance_id / "report.json"
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)
            else:
                results = {"passed": False, "error": "No results file generated"}
            
            # Cleanup
            container.remove()
            
            results["test_output"] = logs
            return results
            
        except Exception as e:
            logger.error(f"Docker validation failed: {str(e)}")
            return {
                "passed": False,
                "error": str(e),
                "test_output": ""
            }

    def _save_logs(self, instance_id: str, result: Dict):
        """Save validation logs in submission format"""
        try:
            # Create logs directory
            log_dir = self.submission_dir / "logs" / instance_id
            os.makedirs(log_dir, exist_ok=True)
            
            # Save test output
            with open(log_dir / "test_output.txt", "w") as f:
                f.write(result["test_output"])
            
            # Save report
            with open(log_dir / "report.json", "w") as f:
                json.dump(result, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save logs for {instance_id}: {str(e)}")

    def save_predictions(self, results: List[ValidationResult]):
        """Save predictions in submission format"""
        try:
            predictions = [
                {
                    "instance_id": r.instance_id,
                    "passed": r.passed,
                    "test_output": r.test_output
                }
                for r in results
            ]
            
            with open(self.submission_dir / "all_preds.jsonl", "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
                    
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}")

    def create_metadata(self, name: str, is_open_source: bool, site_url: str):
        """Create metadata.yaml for submission"""
        try:
            metadata = {
                "name": name,
                "oss": is_open_source,
                "site": site_url,
                "verified": False
            }
            
            with open(self.submission_dir / "metadata.yaml", "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create metadata: {str(e)}")

def main():
    """Example usage"""
    validator = SWEValidator()
    
    # Example validation
    result = validator.validate_solution(
        instance_id="test-1",
        patch_content="Example patch",
        reasoning="Reasoning steps..."
    )
    
    # Save results
    validator.save_predictions([result])
    validator.create_metadata(
        name="MyAgent",
        is_open_source=True,
        site_url="https://github.com/myuser/myagent"
    )

if __name__ == "__main__":
    main()
