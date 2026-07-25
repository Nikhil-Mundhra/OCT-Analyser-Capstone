import argparse
import subprocess
import sys
import os
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Model Health Diagnostic Gauntlet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint (.pth) to diagnose")
    parser.add_argument("--output", type=str, required=True, help="Path to save the markdown report")
    return parser.parse_args()

def run_test(test_name: str, script_path: str, checkpoint: str):
    print(f"Running test: {test_name}...")
    
    # We pass the checkpoint path to the script if it accepts it.
    # Some legacy diagnostic scripts might be hardcoded, so we pass it as an env var 
    # and as a positional argument just in case.
    env = os.environ.copy()
    env["CHECKPOINT_PATH"] = checkpoint
    
    try:
        # We assume the scripts are run from the project root
        result = subprocess.run(
            [sys.executable, script_path, checkpoint],
            capture_output=True,
            text=True,
            env=env,
            timeout=300
        )
        success = result.returncode == 0
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        return success, output
    except Exception as e:
        return False, str(e)

def main():
    args = get_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    os.chdir(project_root)
    
    tests = [
        ("Weight Health Check", "tests/diagnostics/test_weights.py"),
        ("Tensor Scaling Check", "tests/diagnostics/test_div255.py"),
        ("Bias & Features Check", "tests/diagnostics/test_features.py"),
        ("Grad-CAM Explainability Check", "tests/diagnostics/test_cam.py")
    ]
    
    report_lines = [
        f"# Model Health Diagnostics Report",
        f"**Target Checkpoint:** `{args.checkpoint}`\n"
    ]
    
    for name, script_path in tests:
        if not os.path.exists(script_path):
            report_lines.append(f"## {name} \n> [!WARNING]\n> Test script `{script_path}` not found in project.\n")
            continue
            
        success, output = run_test(name, script_path, checkpoint_path)
        
        status_icon = "✅ PASSED" if success else "❌ FAILED"
        report_lines.append(f"## {name} - {status_icon}")
        
        # Wrap output in code block
        report_lines.append("```text")
        report_lines.append(output.strip() if output.strip() else "No output.")
        report_lines.append("```\n")
        
    report_content = "\n".join(report_lines)
    
    out_path = Path(args.output).resolve()
    with open(out_path, "w") as f:
        f.write(report_content)
        
    print(f"Diagnostic gauntlet complete! Report saved to {out_path}")
    print(f"Agent: Please use view_file on {out_path} to read the results.")

if __name__ == "__main__":
    main()
