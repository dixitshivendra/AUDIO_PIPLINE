"""initial_split_schema

Revision ID: 001_initial
Revises: 
Create Date: 2026-06-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # 1. Create Enums explicitly
    job_status = postgresql.ENUM('pending', 'processing', 'completed', 'quarantine', 'failed', name='job_status_enum')
    job_status.create(op.get_bind(), checkfirst=True)
    
    review_status = postgresql.ENUM('pending_review', 'escalated', 'resolved_false_alarm', 'resolved_dispatched', name='review_status_enum')
    review_status.create(op.get_bind(), checkfirst=True)

    # 2. Create tables (with create_type=False to prevent double-creation)
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'processing', 'completed', 'quarantine', 'failed', name='job_status_enum', create_type=False), nullable=False, server_default='pending'),
        sa.Column('extracted_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_detail', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'quarantine_records',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('transcript_excerpt', sa.String(), nullable=True),
        sa.Column('raw_audio_key', sa.String(), nullable=False),
        sa.Column('clean_audio_key', sa.String(), nullable=True),
        sa.Column('review_status', postgresql.ENUM('pending_review', 'escalated', 'resolved_false_alarm', 'resolved_dispatched', name='review_status_enum', create_type=False), nullable=False, server_default='pending_review'),
        sa.Column('reviewer_id', sa.String(), nullable=True),
        sa.Column('reviewer_notes', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id')
    )

def downgrade() -> None:
    op.drop_table('quarantine_records')
    op.drop_table('jobs')
    postgresql.ENUM(name='review_status_enum').drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name='job_status_enum').drop(op.get_bind(), checkfirst=True)
