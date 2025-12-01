#!/bin/bash

echo "🗄️  Setting up SQLite database for ThinkLife..."

# Create .env.local if it doesn't exist
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local file..."
    cat > .env.local << EOL
# Database - SQLite for local development
DATABASE_URL="file:./dev.db"

# Supabase (replace with your actual values if you have them)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here
NEXT_PUBLIC_APP_URL=http://localhost:3000

# NextAuth Secret (generate a random string)
NEXTAUTH_SECRET=your_nextauth_secret_here
EOL
    echo "✅ .env.local created with SQLite configuration"
else
    echo "ℹ️  .env.local already exists"
fi

# Generate Prisma client
echo "🔧 Generating Prisma client..."
npx prisma generate

# Run database migration
echo "🗄️  Running database migrations..."
npx prisma migrate dev --name init

# Set up admin user
echo "👤 Setting up admin user..."
node scripts/setup-admin.js

echo "🎉 Setup complete! Your database is ready and you're set as admin."
echo "🚀 Run 'npm run dev' to start the application."
