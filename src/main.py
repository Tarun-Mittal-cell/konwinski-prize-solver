import os
import logging
from pathlib import Path
from typing import Optional, Dict
import json
import traceback
from agent import GithubIssueAgent, Issue, Solution
from validator import SWEValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SWEBenchRunner:
    """Main runner class integrating agent and validator"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-13b-hf",
        workspace_dir: str = "data/workspace",
        submission_name: str = "CodeLlamaFixer"
    ):
        self.model_name = model_name
        self.workspace_dir = Path(workspace_dir)
        self.submission_name = submission_name
        
        # Initialize components
        self.agent = None
        self.validator = None
        self.results = []
        
    def setup(self) -> None:
        """Initialize all components"""
        try:
            logger.info("Setting up runner...")
            
            # Create directories
            os.makedirs(self.workspace_dir, exist_ok=True)
            
            # Initialize agent
            self.agent = GithubIssueAgent(model_name=self.model_name)
            self.agent.setup()
            
            # Initialize validator
            self.validator = SWEValidator(workspace_dir=str(self.workspace_dir))
            
            logger.info("Setup complete")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def process_dataset(
        self,
        split: str = 'test',
        max_issues: Optional[int] = None
    ) -> None:
        """Process entire dataset"""
        try:
            # Load issues
            issues = self.agent.load_dataset(split)
            if max_issues:
                issues = issues[:max_issues]
                
            logger.info(f"Processing {len(issues)} issues")
            
            # Process each issue
            for idx, issue in enumerate(issues):
                logger.info(f"Processing issue {idx+1}/{len(issues)}: {issue.id}")
                try:
                    self._process_single_issue(issue)
                except Exception as e:
                    logger.error(f"Failed to process issue {issue.id}: {str(e)}")
                    continue
                    
            # Save final results
            self._save_results()
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _process_single_issue(self, issue: Issue) -> None:
        """Process a single issue"""
        try:
            # Generate solution
            solution = self.agent.generate_fix(issue)
            if not solution:
                logger.info(f"No solution generated for {issue.id}")
                return
                
            # Create patch content
            patch_content = self._create_patch(solution.files_modified)
            
            # Create reasoning trace
            reasoning = self._create_reasoning_trace(issue, solution)
            
            # Validate solution
            result = self.validator.validate_solution(
                instance_id=issue.id,
                patch_content=patch_content,
                reasoning=reasoning
            )
            
            self.results.append(result)
            
            # Log result
            status = "PASSED" if result.passed else "FAILED"
            logger.info(f"Issue {issue.id}: {status}")
            
        except Exception as e:
            logger.error(f"Issue processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _create_patch(self, files_modified: Dict[str, str]) -> str:
        """Create patch content from modified files"""
        try:
            patch_lines = []
            for filename, content in files_modified.items():
                patch_lines.extend([
                    f"--- a/{filename}",
                    f"+++ b/{filename}",
                    "@@ -1,1 +1,1 @@",
                    content
                ])
            return "\n".join(patch_lines)
        except Exception as e:
            logger.error(f"Patch creation failed: {str(e)}")
            return ""
            
    def _create_reasoning_trace(self, issue: Issue, solution: Solution) -> str:
        """Create detailed reasoning trace"""
        try:
            trace = [
                "# Solution Reasoning Trace",
                "",
                f"## Issue Analysis",
                f"- Repository: {issue.repo}",
                f"- Issue Title: {issue.title}",
                "",
                "## Proposed Changes",
                "Modified files:",
                *[f"- {f}" for f in solution.files_modified.keys()],
                "",
                "## Confidence Analysis",
                f"- Overall confidence: {solution.confidence:.2f}",
                "",
                "## Implementation Details",
                "Changes made:",
            ]
            
            for filename, content in solution.files_modified.items():
                trace.extend([
                    f"\n### {filename}",
                    "```python",
                    content,
                    "```"
                ])
                
            return "\n".join(trace)
            
        except Exception as e:
            logger.error(f"Reasoning trace creation failed: {str(e)}")
            return "Failed to generate reasoning trace"
            
    def _save_results(self) -> None:
        """Save all results and create submission"""
        try:
            # Save predictions
            self.validator.save_predictions(self.results)
            
            # Create metadata
            self.validator.create_metadata(
                name=self.submission_name,
                is_open_source=True,
                site_url="https://github.com/yourusername/konwinski-prize-solver"
            )
            
            # Save summary
            total = len(self.results)
            passed = sum(1 for r in self.results if r.passed)
            
            summary = {
                "total_issues": total,
                "passed": passed,
                "success_rate": passed / total if total > 0 else 0
            }
            
            with open(self.workspace_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Results saved. Success rate: {summary['success_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    """Run the full pipeline"""
    try:
        # Initialize runner
        runner = SWEBenchRunner()
        runner.setup()
        
        # Process dataset
        runner.process_dataset(
            split='test',
            max_issues=None  # Process all issues
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
