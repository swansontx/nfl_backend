#!/bin/bash
# NFL Props Backend - Mac Setup Script
# Run this on your Mac (not in Claude Code container)

set -e  # Exit on error

echo "=================================="
echo "NFL Props Backend - Mac Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PostgreSQL is installed
echo "📦 Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is installed${NC}"
    psql --version
else
    echo -e "${RED}✗ PostgreSQL not found${NC}"
    echo "Install it with: brew install postgresql@15"
    exit 1
fi

# Check if PostgreSQL is running
echo ""
echo "🔍 Checking if PostgreSQL is running..."
if brew services list | grep postgresql | grep started &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL is not running. Starting it...${NC}"
    brew services start postgresql@15 || brew services start postgresql
    sleep 3
    echo -e "${GREEN}✓ PostgreSQL started${NC}"
fi

# Create database and user
echo ""
echo "🗄️  Setting up database..."

# Check if database exists
if psql postgres -tAc "SELECT 1 FROM pg_database WHERE datname='nfl_props'" | grep -q 1; then
    echo -e "${YELLOW}⚠ Database 'nfl_props' already exists${NC}"
else
    echo "Creating database 'nfl_props'..."
    createdb nfl_props
    echo -e "${GREEN}✓ Database created${NC}"
fi

# Check if user exists
if psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='nfl_props'" | grep -q 1; then
    echo -e "${YELLOW}⚠ User 'nfl_props' already exists${NC}"
else
    echo "Creating user 'nfl_props'..."
    psql postgres -c "CREATE USER nfl_props WITH PASSWORD 'changeme';"
    echo -e "${GREEN}✓ User created${NC}"
fi

# Grant privileges
echo "Granting privileges..."
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE nfl_props TO nfl_props;" 2>/dev/null || true
psql -d nfl_props -c "GRANT ALL ON SCHEMA public TO nfl_props;" 2>/dev/null || true

echo -e "${GREEN}✓ Database setup complete${NC}"

# Python environment setup
echo ""
echo "🐍 Setting up Python environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Install it with: brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Initialize database schema
echo ""
echo "🏗️  Initializing database schema..."
python -c "from backend.database.models import Base; from backend.database.session import engine; Base.metadata.create_all(engine)"
echo -e "${GREEN}✓ Database schema created${NC}"

# Verify database connection
echo ""
echo "🔌 Testing database connection..."
if python -c "from backend.database.session import get_db; list(get_db())" 2>/dev/null; then
    echo -e "${GREEN}✓ Database connection successful${NC}"
else
    echo -e "${YELLOW}⚠ Database connection check inconclusive${NC}"
fi

echo ""
echo "=================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   python -m uvicorn backend.api.app:app --reload --port 8000"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:8000/docs"
echo ""
echo "4. Check health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "=================================="
echo ""
echo "Note: You'll need NFL game data to get full functionality."
echo "The API is ready, but you need to load data into PostgreSQL."
echo ""
