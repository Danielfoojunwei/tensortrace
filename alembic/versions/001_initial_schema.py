"""Initial schema for TensorGuard Platform

Creates all existing tables from the platform models.

Revision ID: 001_initial
Revises: None
Create Date: 2026-01-13
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tenant table
    op.create_table(
        'tenant',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('plan', sa.String(), nullable=False, server_default='starter'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_tenant_name', 'tenant', ['name'])

    # User table
    op.create_table(
        'user',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='operator'),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index('ix_user_email', 'user', ['email'])
    op.create_index('ix_user_tenant_id', 'user', ['tenant_id'])

    # Fleet table
    op.create_table(
        'fleet',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('api_key_hash', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('region', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id', 'name', name='uq_fleet_tenant_name')
    )
    op.create_index('ix_fleet_name', 'fleet', ['name'])
    op.create_index('ix_fleet_tenant_id', 'fleet', ['tenant_id'])
    op.create_index('ix_fleet_region', 'fleet', ['region'])
    op.create_index('ix_fleet_tenant_active', 'fleet', ['tenant_id', 'is_active'])

    # Job table
    op.create_table(
        'job',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('config_json', sa.String(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_job_fleet_id', 'job', ['fleet_id'])
    op.create_index('ix_job_type', 'job', ['type'])
    op.create_index('ix_job_status', 'job', ['status'])
    op.create_index('ix_job_created_at', 'job', ['created_at'])
    op.create_index('ix_job_fleet_status', 'job', ['fleet_id', 'status'])
    op.create_index('ix_job_status_created', 'job', ['status', 'created_at'])

    # AuditLog table
    op.create_table(
        'auditlog',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('resource_id', sa.String(), nullable=False),
        sa.Column('resource_type', sa.String(), nullable=False),
        sa.Column('details', sa.String(), nullable=False, server_default='{}'),
        sa.Column('pqc_signature', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['user_id'], ['user.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_auditlog_tenant_id', 'auditlog', ['tenant_id'])

    # ReplayNonce table
    op.create_table(
        'replaynonce',
        sa.Column('nonce', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.Integer(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('nonce')
    )
    op.create_index('ix_replaynonce_fleet_id', 'replaynonce', ['fleet_id'])
    op.create_index('ix_replaynonce_timestamp', 'replaynonce', ['timestamp'])
    op.create_index('ix_replaynonce_expires_at', 'replaynonce', ['expires_at'])

    # SystemSetting table
    op.create_table(
        'systemsetting',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', sa.String(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('updated_by', sa.String(), nullable=True),
        sa.Column('tenant_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )
    op.create_index('ix_systemsetting_key', 'systemsetting', ['key'])

    # EdgeNode table (existing from settings_models)
    op.create_table(
        'edgenode',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('node_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('fleet_id', sa.String(), nullable=True),
        sa.Column('gating_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('local_threshold', sa.Float(), nullable=False, server_default='0.15'),
        sa.Column('task_whitelist', sa.String(), nullable=False, server_default='[]'),
        sa.Column('status', sa.String(), nullable=False, server_default='offline'),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=True),
        sa.Column('last_ip_address', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.ForeignKeyConstraint(['fleet_id'], ['fleet.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('node_id')
    )
    op.create_index('ix_edgenode_tenant_status', 'edgenode', ['tenant_id', 'status'])
    op.create_index('ix_edgenode_fleet_id', 'edgenode', ['fleet_id'])

    # TelemetrySample table (existing from settings_models - for edge gating)
    op.create_table(
        'telemetrysample',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('node_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('task', sa.String(), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('decision', sa.String(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_telemetrysample_node_timestamp', 'telemetrysample', ['node_id', 'timestamp'])
    op.create_index('ix_telemetrysample_tenant_timestamp', 'telemetrysample', ['tenant_id', 'timestamp'])
    op.create_index('ix_telemetrysample_task', 'telemetrysample', ['task'])

    # GatingDecisionLog table (existing from settings_models)
    op.create_table(
        'gatingdecisionlog',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('node_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('window_start', sa.DateTime(), nullable=False),
        sa.Column('window_end', sa.DateTime(), nullable=False),
        sa.Column('total_decisions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('pass_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('block_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_relevance_score', sa.Float(), nullable=True),
        sa.Column('avg_latency_ms', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenant.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_gatingdecisionlog_node_window', 'gatingdecisionlog', ['node_id', 'window_start'])


def downgrade() -> None:
    op.drop_table('gatingdecisionlog')
    op.drop_table('telemetrysample')
    op.drop_table('edgenode')
    op.drop_table('systemsetting')
    op.drop_table('replaynonce')
    op.drop_table('auditlog')
    op.drop_table('job')
    op.drop_table('fleet')
    op.drop_table('user')
    op.drop_table('tenant')
