# User Database Design

## Schema
- **users**
  - id (PK)
  - name
  - email (unique)
  - password_hash
  - role
  - created_at

## Relationships
- One-to-Many with `credit_scores`
- One-to-Many with `uploads`

## Notes
- Use PostgreSQL in production
- SQLite allowed for testing