"""Add FedMoE expert registry tables

Revision ID: 003_fedmoe_experts
Revises: 002_telemetry_and_rollout
Create Date: 2026-01-14
"""
from alembic import op
import sqlalchemy as sa

revision = "003_fedmoe_experts"
down_revision = "002_telemetry_and_rollout"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fedmoeexpert",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("base_model", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False, server_default="1.0.0"),
        sa.Column("status", sa.String(), nullable=False, server_default="adapting"),
        sa.Column("accuracy_score", sa.Float(), nullable=True),
        sa.Column("collision_rate", sa.Float(), nullable=True),
        sa.Column("success_rate", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("safety_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("gating_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenant.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", "version", name="uq_expert_tenant_name_version"),
    )
    op.create_index("ix_expert_tenant_status", "fedmoeexpert", ["tenant_id", "status"])
    op.create_index("ix_expert_base_model", "fedmoeexpert", ["base_model"])
    op.create_index("ix_fedmoeexpert_tenant_id", "fedmoeexpert", ["tenant_id"])
    op.create_index("ix_fedmoeexpert_name", "fedmoeexpert", ["name"])

    op.create_table(
        "skillevidence",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("expert_id", sa.String(), nullable=False),
        sa.Column("evidence_type", sa.String(), nullable=False),
        sa.Column("value_json", sa.String(), nullable=False, server_default="{}"),
        sa.Column("signed_proof", sa.String(), nullable=True),
        sa.Column("manifest_hash", sa.String(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["expert_id"], ["fedmoeexpert.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_skillevidence_expert_id", "skillevidence", ["expert_id"])


def downgrade() -> None:
    op.drop_index("ix_skillevidence_expert_id", table_name="skillevidence")
    op.drop_table("skillevidence")
    op.drop_index("ix_fedmoeexpert_name", table_name="fedmoeexpert")
    op.drop_index("ix_fedmoeexpert_tenant_id", table_name="fedmoeexpert")
    op.drop_index("ix_expert_base_model", table_name="fedmoeexpert")
    op.drop_index("ix_expert_tenant_status", table_name="fedmoeexpert")
    op.drop_table("fedmoeexpert")
