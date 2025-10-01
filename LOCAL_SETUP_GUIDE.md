# ThinkxLife Local Development Setup Guide

This guide will help you set up and run ThinkxLife locally on your machine, even if you don't have Node.js or Python installed.

## Prerequisites Installation

### 1. Install Node.js (Required for Frontend)

#### Windows:
1. Visit [nodejs.org](https://nodejs.org/)
2. Download the LTS version (recommended)
3. Run the installer and follow the setup wizard
4. Restart your terminal/command prompt

#### macOS:
```bash
# Option 1: Direct download
# Visit https://nodejs.org/ and download the LTS version

# Option 2: Using Homebrew (if you have it)
brew install node

# Option 3: Using MacPorts
sudo port install nodejs18
```

#### Linux (Ubuntu/Debian):
```bash
# Update package index
sudo apt update

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

### 2. Install Python (Required for Backend)

#### Windows:
1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.8 or higher
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Run the installer

#### macOS:
```bash
# Option 1: Direct download from python.org
# Visit https://www.python.org/downloads/

# Option 2: Using Homebrew
brew install python

# Option 3: Using pyenv (recommended for version management)
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 3. Install Git (if not already installed)

#### Windows:
Download from [git-scm.com](https://git-scm.com/download/win)

#### macOS:
```bash
# Usually pre-installed, or install via Homebrew
brew install git
```

#### Linux:
```bash
sudo apt install git
```

## Project Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ThinkxLife
```

### Step 2: Backend Setup (Python)

#### Navigate to Backend Directory
```bash
cd backend
```

#### Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Create Backend Environment File
Create a file named `.env` in the `backend/` directory:

```bash
# Create .env file (Windows)
echo. > .env

# Create .env file (macOS/Linux)
touch .env
```

Add the following content to `backend/.env`:
```env
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.7

# Vector Database Configuration (OPTIONAL)
CONTEXT_TXT_PATH=data/context.txt
KNOWLEDGE_JSON_PATH=data/knowledge_base.json
EMPATHY_CSV_PATH=data/empathetic_dialogues_train.csv
CHUNK_SIZE=300
MAX_EMPATHY_EXAMPLES=500
CHROMA_DB_DIR=chroma_db

# Gemini Configuration (OPTIONAL)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_TOKENS=2000
GEMINI_TEMPERATURE=0.7

#web search functionality (OPTIONAL)
TAVILY_API_KEY=your_tavily_api_key_here

# Server Configuration
PORT=8000
DEBUG=True

# CORS Settings
CORS_ORIGINS=http://localhost:3000

```

### Step 3: Frontend Setup (Node.js)

#### Navigate to Frontend Directory
```bash
cd ../frontend
```

#### Install Node.js Dependencies
```bash
npm install
```

#### Create Frontend Environment File
Create a file named `.env.local` in the `frontend/` directory:

```bash
# Create .env.local file (Windows)
echo. > .env.local

# Create .env.local file (macOS/Linux)
touch .env.local
```

Add the following content to `frontend/.env.local`:
```env
# Next.js Configuration (REQUIRED)
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your_nextauth_secret_here_minimum_32_characters_long

# Database Configuration (OPTIONAL - for user management features)
DATABASE_URL="postgresql://username:password@localhost:5432/thinkxlife"
SHADOW_DATABASE_URL="postgresql://username:password@localhost:5432/thinkxlife_shadow"

# External API Endpoints (REQUIRED)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_URL=http://localhost:3000

# Supabase Configuration (REQUIRED for authentication)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Admin Configuration (OPTIONAL)
ADMIN_EMAILS=admin@example.com,admin2@example.com
NEXT_PUBLIC_ADMIN_EMAILS=admin@example.com,admin2@example.com

# Email Configuration (OPTIONAL - for password reset)
RESEND_API_KEY=your_resend_api_key_here

# Google OAuth Configuration (OPTIONAL)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

```

#### Generate NextAuth Secret
```bash
# Generate a secure secret for NextAuth
node scripts/generate-auth-secret.js
```

### Step 4: Database Setup (Prisma)

#### Initialize Database Schema
```bash
# Generate Prisma client
npx prisma generate

# Push database schema (for development)
npx prisma db push

# OR run migrations (for production-like setup)
npx prisma migrate dev --name init

# (Optional) Seed database with initial data
npx prisma db seed
```

#### View Database (Optional)
```bash
# Open Prisma Studio to view/edit data
npx prisma studio
```

## Running the Application

### Option 1: Run Both Services Simultaneously

#### Terminal 1 - Backend Server:
```bash
cd backend

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Start backend server
python3 main.py
# OR
python3 start.py
```

#### Terminal 2 - Frontend Server:
```bash
cd frontend

# Start frontend development server
npm run dev
```

### Option 2: Using Scripts (if available)

Some projects have convenience scripts. Check if these work:
```bash
# From project root
npm run dev        # If there's a root package.json
npm run start:dev  # Alternative script name
```

## Accessing the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Prisma Studio** (if running): http://localhost:5555

## Troubleshooting

### Common Issues and Solutions

#### Node.js Issues:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules
npm install

# Check Node.js version (should be 18+ for compatibility)
node --version
```

#### Python Issues:
```bash
# Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### Database Issues:
```bash
# Reset database
npx prisma migrate reset

# Regenerate Prisma client
npx prisma generate

# Push schema changes
npx prisma db push
```

#### Environment File Issues:
- Ensure `.env` files are in the correct directories
- Check that all required variables are set
- Verify no spaces around the `=` sign in env files
- Ensure sensitive values are properly quoted if they contain special characters

#### Port Conflicts:
If ports 3000 or 8000 are already in use:

**Frontend (Next.js):**
```bash
npm run dev -- -p 3001  # Run on port 3001
```

**Backend:**
Update the PORT variable in `backend/.env` or run with:
```bash
PORT=8001 python main.py
```

### Getting API Keys

#### OpenAI API Key:
1. Visit [platform.openai.com](https://platform.openai.com/)
2. Sign up/log in
3. Go to API Keys section
4. Create a new secret key
5. Add to `backend/.env`

#### Google OAuth (if needed):
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add to `frontend/.env.local`

## Development Workflow

### Making Changes:
1. Backend changes: Server auto-reloads if using development mode
2. Frontend changes: Next.js hot-reloads automatically
3. Database changes: Run `npx prisma db push` or create migrations

### Stopping the Application:
- Press `Ctrl+C` in both terminal windows
- Deactivate Python virtual environment: `deactivate`

## Production Deployment Notes

This guide is for local development. For production:
- Use proper database (PostgreSQL, MySQL)
- Set `NODE_ENV=production`
- Use proper secret keys
- Configure proper CORS origins
- Use process managers (PM2, systemd)
- Set up reverse proxy (nginx)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all prerequisites are properly installed
3. Verify environment files are correctly configured
4. Check the project's GitHub issues
5. Review application logs in the terminal

---

**Note**: Replace placeholder values (like `your_openai_api_key_here`) with actual values for your setup. 