import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .models.evidence_models import PolicyPack, RunPolicyResult
from .database import engine
from sqlmodel import Session, select


def _get_default_packs_dir() -> str:
    """Get the default policy packs directory using absolute path resolution."""
    # Resolve from module location: src/tensorguard/platform -> project root
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent.parent.parent
    return str(project_root / "configs" / "policy_packs")


class PolicyEngine:
    def __init__(self, packs_dir: str = None):
        if packs_dir is None:
            self.packs_dir = _get_default_packs_dir()
        else:
            self.packs_dir = packs_dir
            
        self.packs = {}
        self.load_packs()

    def load_packs(self):
        """Load YAML packs and sync with DB."""
        if not os.path.exists(self.packs_dir):
            return

        with Session(engine) as session:
            for f in os.listdir(self.packs_dir):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    path = os.path.join(self.packs_dir, f)
                    with open(path, 'r') as fh:
                        data = yaml.safe_load(fh)
                        self.packs[data['id']] = data
                        
                        # Sync to DB
                        pack = session.exec(select(PolicyPack).where(PolicyPack.id == data['id'])).first()
                        if not pack:
                            pack = PolicyPack(
                                id=data['id'], 
                                name=data['name'], 
                                description=data['description'],
                                version=str(data.get('version', '1.0.0')),
                                hash="local-file" # Placeholder for local sync
                            )
                            session.add(pack)
                        else:
                            pack.version = str(data.get('version', '1.0.0'))
                            session.add(pack)
            session.commit()

    def evaluate(self, run_id: str, report_json: Dict[str, Any], pack_id: str = "soc2-evidence-pack") -> RunPolicyResult:
        if pack_id not in self.packs:
            raise ValueError(f"Policy pack {pack_id} not found")
            
        pack = self.packs[pack_id]
        score = 0.0
        max_score = sum(r['weight'] for r in pack['rules'])
        reasons = []
        passed_all = True
        
        # Flatten metrics for easier access
        # report.json structure: "metrics": {"privacy_mse": ...}
        # Flat dict: "metrics.privacy_mse"
        metrics = report_json.get('metrics', {}) or {}
        
        for rule in pack['rules']:
            metric_path = rule['metric']
            # Simple accessor: metrics.key
            key = metric_path.split('.')[-1]
            val = metrics.get(key)
            
            rule_passed = False
            
            if val is None:
                reasons.append(f"Missing metric: {key}")
                passed_all = False
                continue
                
            op = rule['operator']
            threshold = rule['value']
            
            if op == 'lt':
                rule_passed = val < threshold
            elif op == 'gt':
                rule_passed = val > threshold
            elif op == 'eq':
                rule_passed = val == threshold
            elif op == 'lte':
                rule_passed = val <= threshold
            elif op == 'gte':
                rule_passed = val >= threshold
            
            if rule_passed:
                score += rule['weight']
            else:
                passed_all = False
                msg = rule['failure_msg'].format(value=val, threshold=threshold)
                reasons.append(msg)

        final_score = (score / max_score) * 100.0 if max_score > 0 else 0.0
        
        result = RunPolicyResult(
            run_id=run_id,
            pack_id=pack_id,
            pack_version=str(pack['version']),
            passed=passed_all,
            score=final_score,
            reasons_json=json.dumps(reasons)
        )
        return result
