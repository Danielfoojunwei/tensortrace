from sqlmodel import Session, select
from datetime import datetime
import json
import uuid

from .database import engine, init_db
from .models.fedmoe_models import FedMoEExpert, SkillEvidence
from .models.core import Tenant, User

def seed_fedmoe():
    init_db() # Ensure tables exist
    with Session(engine) as session:
        # Get first tenant or create one
        tenant = session.exec(select(Tenant)).first()
        if not tenant:
            print("Creating default tenant...")
            tenant = Tenant(name="Enterprise Demo", plan="Enterprise")
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            
            user = User(
                email="admin@tensorguard.ai",
                hashed_password="N/A", # Dummy
                role="org_admin",
                tenant_id=tenant.id
            )
            session.add(user)
            session.commit()
            
        # 1. Expert: Manipulation Grasp
        expert1 = FedMoEExpert(
            name="manipulation_grasp_v4",
            base_model="openvla-7b",
            tenant_id=tenant.id,
            status="validated",
            accuracy_score=0.94,
            collision_rate=0.002
        )
        session.add(expert1)
        session.commit()
        session.refresh(expert1)
        
        # Add Evidence
        ev1 = SkillEvidence(
            expert_id=expert1.id,
            evidence_type="SIM_SUCCESS",
            value_json=json.dumps({"score": 0.94, "collision_rate": 0.002}),
            signed_proof="dilithium3_sig_" + str(uuid.uuid4())[:8]
        )
        session.add(ev1)
        
        # 2. Expert: Cloth Folding
        expert2 = FedMoEExpert(
            name="cloth_folding_lora",
            base_model="pi-0",
            tenant_id=tenant.id,
            status="adapting"
        )
        session.add(expert2)
        
        # 3. Expert: Drawer Operation
        expert3 = FedMoEExpert(
            name="drawer_op_v1",
            base_model="openvla-7b",
            tenant_id=tenant.id,
            status="validated",
            accuracy_score=0.88,
            collision_rate=0.012
        )
        session.add(expert3)
        session.commit()
        session.refresh(expert3)
        
        ev2 = SkillEvidence(
            expert_id=expert3.id,
            evidence_type="REAL_WORLD_VLD",
            value_json=json.dumps({"score": 0.88, "location": "Site-B"}),
            signed_proof="dilithium3_sig_" + str(uuid.uuid4())[:8]
        )
        session.add(ev2)
        
        session.commit()
        print("FedMoE data seeded successfully.")

if __name__ == "__main__":
    seed_fedmoe()
