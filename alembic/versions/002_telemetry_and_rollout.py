"""Add telemetry and rollout tables for production control plane

Creates new tables:
- fleet_device: Device registry with version tracking
- telemetry_stage_event: Pipeline stage telemetry
- telemetry_system_event: System resource telemetry
- telemetry_model_behavior_event: Model decision telemetry (shadow/A-B)
- forensics_event: Security and safety events
- telemetry_retention_policy: Data retention policies
- deployment_plan: Deployment rollout plans
- deployment_assignment: Device deployment assignments
- rollback_event: Rollback trigger events

Revision ID: 002_telemetry_rollout
Revises: 001_initial
Create Date: 2026-01-13
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_telemetry_rollout'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ===========================================================================
    # FleetDevice - Device Registry
    # ===========================================================================
    op.create_table(
        'fleet_device',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('agent_version', sa.String(), nullable=True),
        sa.Column('runtime_version', sa.String(), nullable=True),
        sa.Column('ros_distro', sa.String(), nullable=True),
        sa.Column('firmware_version', sa.String(), nullable=True),
        sa.Column('sensor_manifest_hash', sa.String(), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('device_id')
    )
    op.create_index('ix_fleet_device_device_id', 'fleet_device', ['device_id'])
    op.create_index('ix_fleet_device_tenant_id', 'fleet_device', ['tenant_id'])
    op.create_index('ix_fleet_device_fleet_id', 'fleet_device', ['fleet_id'])
    op.create_index('ix_fleet_device_last_seen_at', 'fleet_device', ['last_seen_at'])
    op.create_index('ix_fleet_device_tenant_fleet', 'fleet_device', ['tenant_id', 'fleet_id'])
    op.create_index('ix_fleet_device_fleet_lastseen', 'fleet_device', ['fleet_id', 'last_seen_at'])

    # ===========================================================================
    # TelemetryStageEvent - Pipeline Stage Telemetry
    # ===========================================================================
    op.create_table(
        'telemetry_stage_event',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('run_id', sa.String(), nullable=True),
        sa.Column('stage', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='ok'),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('metadata_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_telemetry_stage_tenant_id', 'telemetry_stage_event', ['tenant_id'])
    op.create_index('ix_telemetry_stage_fleet_id', 'telemetry_stage_event', ['fleet_id'])
    op.create_index('ix_telemetry_stage_device_id', 'telemetry_stage_event', ['device_id'])
    op.create_index('ix_telemetry_stage_run_id', 'telemetry_stage_event', ['run_id'])
    op.create_index('ix_telemetry_stage_stage', 'telemetry_stage_event', ['stage'])
    op.create_index('ix_telemetry_stage_ts', 'telemetry_stage_event', ['ts'])
    op.create_index('ix_telemetry_stage_fleet_ts', 'telemetry_stage_event', ['fleet_id', 'ts'])
    op.create_index('ix_telemetry_stage_fleet_stage_ts', 'telemetry_stage_event', ['fleet_id', 'stage', 'ts'])
    op.create_index('ix_telemetry_stage_device_ts', 'telemetry_stage_event', ['device_id', 'ts'])
    op.create_index('ix_telemetry_stage_tenant_ts', 'telemetry_stage_event', ['tenant_id', 'ts'])

    # ===========================================================================
    # TelemetrySystemEvent - System Resource Telemetry
    # ===========================================================================
    op.create_table(
        'telemetry_system_event',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('cpu_pct', sa.Float(), nullable=False),
        sa.Column('mem_pct', sa.Float(), nullable=False),
        sa.Column('gpu_pct', sa.Float(), nullable=True),
        sa.Column('temp_c', sa.Float(), nullable=True),
        sa.Column('bandwidth_up_bps', sa.Integer(), nullable=True),
        sa.Column('bandwidth_down_bps', sa.Integer(), nullable=True),
        sa.Column('dropped_frames', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('queue_latency_ms', sa.Float(), nullable=True),
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_telemetry_system_tenant_id', 'telemetry_system_event', ['tenant_id'])
    op.create_index('ix_telemetry_system_fleet_id', 'telemetry_system_event', ['fleet_id'])
    op.create_index('ix_telemetry_system_device_id', 'telemetry_system_event', ['device_id'])
    op.create_index('ix_telemetry_system_ts', 'telemetry_system_event', ['ts'])
    op.create_index('ix_telemetry_system_fleet_ts', 'telemetry_system_event', ['fleet_id', 'ts'])
    op.create_index('ix_telemetry_system_device_ts', 'telemetry_system_event', ['device_id', 'ts'])
    op.create_index('ix_telemetry_system_tenant_ts', 'telemetry_system_event', ['tenant_id', 'ts'])

    # ===========================================================================
    # TelemetryModelBehaviorEvent - Model Decision Telemetry
    # ===========================================================================
    op.create_table(
        'telemetry_model_behavior_event',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('model_version', sa.String(), nullable=False),
        sa.Column('adapter_id', sa.String(), nullable=True),
        sa.Column('decision_hash', sa.String(), nullable=False),
        sa.Column('action_distribution_json', sa.String(), nullable=True),
        sa.Column('refusal_rate', sa.Float(), nullable=True),
        sa.Column('tool_call_failures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('policy_constraint_hits', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_shadow', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_telemetry_model_tenant_id', 'telemetry_model_behavior_event', ['tenant_id'])
    op.create_index('ix_telemetry_model_fleet_id', 'telemetry_model_behavior_event', ['fleet_id'])
    op.create_index('ix_telemetry_model_device_id', 'telemetry_model_behavior_event', ['device_id'])
    op.create_index('ix_telemetry_model_model_version', 'telemetry_model_behavior_event', ['model_version'])
    op.create_index('ix_telemetry_model_adapter_id', 'telemetry_model_behavior_event', ['adapter_id'])
    op.create_index('ix_telemetry_model_is_shadow', 'telemetry_model_behavior_event', ['is_shadow'])
    op.create_index('ix_telemetry_model_ts', 'telemetry_model_behavior_event', ['ts'])
    op.create_index('ix_telemetry_model_fleet_ts', 'telemetry_model_behavior_event', ['fleet_id', 'ts'])
    op.create_index('ix_telemetry_model_device_ts', 'telemetry_model_behavior_event', ['device_id', 'ts'])
    op.create_index('ix_telemetry_model_version_ts', 'telemetry_model_behavior_event', ['model_version', 'ts'])
    op.create_index('ix_telemetry_model_shadow', 'telemetry_model_behavior_event', ['is_shadow', 'ts'])
    op.create_index('ix_telemetry_model_tenant_ts', 'telemetry_model_behavior_event', ['tenant_id', 'ts'])

    # ===========================================================================
    # ForensicsEvent - Security and Safety Events
    # ===========================================================================
    op.create_table(
        'forensics_event',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=True),
        sa.Column('severity', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('evidence_ref', sa.String(), nullable=True),
        sa.Column('details_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_forensics_tenant_id', 'forensics_event', ['tenant_id'])
    op.create_index('ix_forensics_fleet_id', 'forensics_event', ['fleet_id'])
    op.create_index('ix_forensics_device_id', 'forensics_event', ['device_id'])
    op.create_index('ix_forensics_severity', 'forensics_event', ['severity'])
    op.create_index('ix_forensics_event_type', 'forensics_event', ['event_type'])
    op.create_index('ix_forensics_ts', 'forensics_event', ['ts'])
    op.create_index('ix_forensics_fleet_ts', 'forensics_event', ['fleet_id', 'ts'])
    op.create_index('ix_forensics_device_ts', 'forensics_event', ['device_id', 'ts'])
    op.create_index('ix_forensics_severity_ts', 'forensics_event', ['severity', 'ts'])
    op.create_index('ix_forensics_type_ts', 'forensics_event', ['event_type', 'ts'])
    op.create_index('ix_forensics_tenant_ts', 'forensics_event', ['tenant_id', 'ts'])

    # ===========================================================================
    # TelemetryRetentionPolicy - Data Retention Management
    # ===========================================================================
    op.create_table(
        'telemetry_retention_policy',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=True),
        sa.Column('table_name', sa.String(), nullable=False),
        sa.Column('retention_days', sa.Integer(), nullable=False, server_default='30'),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_purge_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('table_name')
    )
    op.create_index('ix_retention_tenant_id', 'telemetry_retention_policy', ['tenant_id'])
    op.create_index('ix_retention_table_name', 'telemetry_retention_policy', ['table_name'])

    # ===========================================================================
    # DeploymentPlan - Rollout Plans
    # ===========================================================================
    op.create_table(
        'deployment_plan',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('target_model_version', sa.String(), nullable=False),
        sa.Column('target_adapter_id', sa.String(), nullable=True),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('stages_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('guardrails_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('compatibility_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('status', sa.String(), nullable=False, server_default='draft'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_deployment_plan_tenant_id', 'deployment_plan', ['tenant_id'])
    op.create_index('ix_deployment_plan_fleet_id', 'deployment_plan', ['fleet_id'])
    op.create_index('ix_deployment_plan_status', 'deployment_plan', ['status'])
    op.create_index('ix_deployment_plan_fleet_status', 'deployment_plan', ['fleet_id', 'status'])
    op.create_index('ix_deployment_plan_tenant_created', 'deployment_plan', ['tenant_id', 'created_at'])

    # ===========================================================================
    # DeploymentAssignment - Device Assignments
    # ===========================================================================
    op.create_table(
        'deployment_assignment',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('deployment_id', sa.String(), nullable=False),
        sa.Column('assigned_variant', sa.String(), nullable=True),
        sa.Column('assigned_adapter_id', sa.String(), nullable=True),
        sa.Column('is_shadow', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.ForeignKeyConstraint(['deployment_id'], ['deployment_plan.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('deployment_id', 'device_id', name='uq_deployment_device')
    )
    op.create_index('ix_deployment_assignment_tenant_id', 'deployment_assignment', ['tenant_id'])
    op.create_index('ix_deployment_assignment_fleet_id', 'deployment_assignment', ['fleet_id'])
    op.create_index('ix_deployment_assignment_device_id', 'deployment_assignment', ['device_id'])
    op.create_index('ix_deployment_assignment_deployment_id', 'deployment_assignment', ['deployment_id'])

    # ===========================================================================
    # RollbackEvent - Rollback Triggers
    # ===========================================================================
    op.create_table(
        'rollback_event',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('deployment_id', sa.String(), nullable=False),
        sa.Column('trigger_type', sa.String(), nullable=False),
        sa.Column('trigger_details_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('ts', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.ForeignKeyConstraint(['deployment_id'], ['deployment_plan.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rollback_event_tenant_id', 'rollback_event', ['tenant_id'])
    op.create_index('ix_rollback_event_fleet_id', 'rollback_event', ['fleet_id'])
    op.create_index('ix_rollback_event_deployment_id', 'rollback_event', ['deployment_id'])
    op.create_index('ix_rollback_event_ts', 'rollback_event', ['ts'])

    # ===========================================================================
    # Insert default retention policies
    # ===========================================================================
    op.execute("""
        INSERT INTO telemetry_retention_policy (id, tenant_id, table_name, retention_days, enabled, created_at, updated_at)
        VALUES
            ('retention-stage', NULL, 'telemetry_stage_event', 30, true, datetime('now'), datetime('now')),
            ('retention-system', NULL, 'telemetry_system_event', 7, true, datetime('now'), datetime('now')),
            ('retention-model', NULL, 'telemetry_model_behavior_event', 90, true, datetime('now'), datetime('now')),
            ('retention-forensics', NULL, 'forensics_event', 365, true, datetime('now'), datetime('now'))
    """)


def downgrade() -> None:
    op.drop_table('rollback_event')
    op.drop_table('deployment_assignment')
    op.drop_table('deployment_plan')
    op.drop_table('telemetry_retention_policy')
    op.drop_table('forensics_event')
    op.drop_table('telemetry_model_behavior_event')
    op.drop_table('telemetry_system_event')
    op.drop_table('telemetry_stage_event')
    op.drop_table('fleet_device')
