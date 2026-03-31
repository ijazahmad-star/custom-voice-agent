"""enable_pgvector_extension

Revision ID: e9c2f68eed9c
Revises: 
Create Date: 2026-03-31 16:33:29.716925

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9c2f68eed9c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
