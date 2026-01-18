
import time
import json
import os
import random
from datetime import datetime
from sqlmodel import Session, select
from tensorguard.platform.database import engine
from tensorguard.platform.models.core import AuditLog, Tenant, User
from tensorguard.platform.models.fedmoe_models import FedMoEExpert, SkillEvidence
from tensorguard.bench.production_demo import ProductionPipelineTracer
from tensorguard.core.production import UpdatePackage

class QASimulation:
    def __init__(self):
        self.tracer = ProductionPipelineTracer(fleet_id="qa_fleet_600")
        self.tenant_id = None
        self.user_id = None

    def setup_db(self):
        with Session(engine) as session:
            # Ensure we have a tenant and user for the simulation
            tenant = session.exec(select(Tenant).where(Tenant.name == "QA_Simulation_Corp")).first()
            if not tenant:
                tenant = Tenant(name="QA_Simulation_Corp", plan="enterprise")
                session.add(tenant)
                session.commit()
            
            user = session.exec(select(User).where(User.email == "sim_bot@qa.com")).first()
            if not user:
                user = User(email="sim_bot@qa.com", hashed_password="mock", role="service_account", tenant_id=tenant.id)
                session.add(user)
                session.commit()
            
            self.tenant_id = tenant.id
            self.user_id = user.id
            print(f"Database setup complete. Tenant: {self.tenant_id}")

    def inject_expert_version(self, task: str, version_idx: int, accuracy: float):
        with Session(engine) as session:
            expert = FedMoEExpert(
                name=f"Expert-{task}",
                base_model="OpenVLA-7B-Quant",
                version=f"v2.{version_idx}",
                status="adapting" if version_idx % 2 == 0 else "deployed",
                accuracy_score=accuracy,
                tenant_id=self.tenant_id,
                created_at=datetime.utcnow()
            )
            session.add(expert)
            session.commit()
            
            # Add Evidence
            ev = SkillEvidence(
                expert_id=expert.id,
                evidence_type="SIM_SUCCESS",
                value_json=json.dumps({"success_rate": accuracy}),
                signed_proof=f"sig_dilithium3_{os.urandom(8).hex()}"
            )
            session.add(ev)
            session.commit()
            print(f"Injected Expert Version: {expert.name} {expert.version}")

    def inject_audit_log(self, action: str, details: dict):
        with Session(engine) as session:
            log = AuditLog(
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                action=action,
                resource_id="sim-run-600",
                resource_type="simulation",
                details=json.dumps(details),
                pqc_signature=f"sig_dilithium3_{os.urandom(4).hex()}",
                timestamp=datetime.utcnow()
            )
            session.add(log)
            session.commit()

    def run(self):
        self.setup_db()
        
        total_cycles = 600
        cycle_tasks = ["task_a"] * 300 + ["task_b"] * 300
        
        print(f"Starting {total_cycles} cycle simulation...")
        
        for i, task in enumerate(cycle_tasks):
            print(f"Cycle {i+1}/{total_cycles} [{task}]")
            
            # 1. Run Core Pipeline Trace (In-Memory)
            self.tracer.run_iteration(i, task)
            
            # 2. Inject Real DB Events for UI
            if i % 10 == 0:
                self.inject_audit_log("GRADIENT_UPDATE", {"cycle": i, "task": task, "latency": random.randint(40, 60)})
            
            if i % 50 == 0:
                acc = 0.85 + (0.0001 * i) # Increasing accuracy
                self.inject_expert_version(task, int(i/50)+1, acc)
                self.inject_audit_log("MODEL_VERSION_BUMP", {"version": f"v2.{int(i/50)+1}", "accuracy": acc})
                
            time.sleep(0.1) # Fast forward simulation

if __name__ == "__main__":
    sim = QASimulation()
    sim.run()
