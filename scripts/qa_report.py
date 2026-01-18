
import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime

ARTIFACTS_DIR = "artifacts"
REPORT_XML = f"{ARTIFACTS_DIR}/junit_report.xml"
SUMMARY_MD = f"{ARTIFACTS_DIR}/qa_summary.md"

def run_tests():
    """Run pytest and generate JUnit XML and Coverage."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--junitxml=" + REPORT_XML,
        # "--cov=src/tensorguard", "--cov-report=html", "--cov-report=term" # Optional if installed
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False) # Let stdout flow
    return result.returncode

def generate_summary():
    """Parse XML and write Markdown summary."""
    if not os.path.exists(REPORT_XML):
        print("No JUnit XML found.")
        return

    tree = ET.parse(REPORT_XML)
    root = tree.getroot()
    
    # Root element is usually <testsuites> or <testsuite>
    if root.tag == "testsuites":
        suites = list(root)
    else:
        suites = [root]
        
    total_tests = 0
    failures = 0
    errors = 0
    skipped = 0
    time_taken = 0.0
    
    failed_cases = []
    
    for suite in suites:
        total_tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        time_taken += float(suite.get("time", 0))
        
        for case in suite.findall("testcase"):
            result = list(case)
            if result and result[0].tag in ["failure", "error"]:
                failed_cases.append({
                    "name": case.get("name"),
                    "class": case.get("classname"),
                    "message": result[0].get("message")
                })

    passed = total_tests - failures - errors - skipped
    pass_rate = (passed / (total_tests - skipped)) * 100 if (total_tests - skipped) > 0 else 0
    
    # Write Markdown
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(f"# TensorGuard QA Summary\n\n")
        f.write(f"- **Date**: {datetime.now().isoformat()}\n")
        f.write(f"- **Duration**: {time_taken:.2f}s\n")
        f.write(f"- **Pass Rate**: {pass_rate:.1f}%\n\n")
        
        f.write(f"| Metric | Count |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Total | {total_tests} |\n")
        f.write(f"| Passed | {passed} |\n")
        f.write(f"| Failed | {failures} |\n")
        f.write(f"| Errors | {errors} |\n")
        f.write(f"| Skipped | {skipped} |\n\n")
        
        if failed_cases:
            f.write("## 🚨 Failed Tests\n\n")
            for case in failed_cases:
                f.write(f"### {case['class']}::{case['name']}\n")
                f.write(f"```text\n{case['message']}\n```\n")
        else:
            f.write("## ✅ All Tests Passed\n")
            
    print(f"Summary written to {SUMMARY_MD}")

if __name__ == "__main__":
    ret = run_tests()
    generate_summary()
    sys.exit(ret)
