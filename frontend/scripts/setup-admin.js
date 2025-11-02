#!/usr/bin/env node

const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

async function setupAdmin() {
  console.log('🚀 Setting up admin user...');
  
  // Check if DATABASE_URL is set
  if (!process.env.DATABASE_URL) {
    console.error('❌ DATABASE_URL environment variable is not set!');
    console.log('\n📝 Please create a .env.local file with:');
    console.log('DATABASE_URL="postgresql://username:password@localhost:5432/thinklife?schema=public"');
    console.log('\nOr for a quick SQLite setup:');
    console.log('DATABASE_URL="file:./dev.db"');
    process.exit(1);
  }

  const prisma = new PrismaClient();

  try {
    console.log('📊 Connecting to database...');
    
    // Check if user already exists
    let user = await prisma.user.findUnique({
      where: { email: 'jagadeshvarma07@gmail.com' }
    });

    if (!user) {
      console.log('👤 Creating user jagadeshvarma07@gmail.com...');
      user = await prisma.user.create({
        data: {
          email: 'jagadeshvarma07@gmail.com',
          name: 'Jagadesh Varma Nadimpalli',
          rolePrimary: 'ADMIN',
          status: 'ACTIVE',
          createdAt: new Date(),
          updatedAt: new Date(),
        }
      });
      console.log('✅ User created successfully!');
    } else {
      console.log('👤 User exists, updating role to ADMIN...');
      user = await prisma.user.update({
        where: { email: 'jagadeshvarma07@gmail.com' },
        data: {
          rolePrimary: 'ADMIN',
          status: 'ACTIVE',
          updatedAt: new Date(),
        }
      });
      console.log('✅ User updated successfully!');
    }

    // Create default roles if they don't exist
    console.log('🏷️  Setting up default roles...');
    
    const roles = [
      { code: 'MEMBER', name: 'Member', description: 'Basic member access' },
      { code: 'INTERN', name: 'Intern', description: 'Intern with limited access' },
      { code: 'TEAM_LEAD', name: 'Team Lead', description: 'Team leadership access' },
      { code: 'ADMIN', name: 'Administrator', description: 'Full system access' },
    ];

    for (const roleData of roles) {
      try {
        await prisma.role.upsert({
          where: { code: roleData.code },
          update: {},
          create: roleData
        });
        console.log(`✅ Role ${roleData.code} set up`);
      } catch (error) {
        console.log(`ℹ️  Role ${roleData.code} may already exist or table doesn't exist yet`);
      }
    }

    // Assign admin role to user if UserRole_Assignment table exists
    try {
      const adminRole = await prisma.role.findUnique({
        where: { code: 'ADMIN' }
      });

      if (adminRole) {
        await prisma.userRole_Assignment.upsert({
          where: {
            userId_roleId: {
              userId: user.id,
              roleId: adminRole.id
            }
          },
          update: {},
          create: {
            userId: user.id,
            roleId: adminRole.id,
            assignedBy: user.id,
            assignedAt: new Date(),
          }
        });
        console.log('✅ Admin role assigned to user');
      }
    } catch (error) {
      console.log('ℹ️  Role assignment table may not exist yet, but user has rolePrimary set to ADMIN');
    }

    console.log('\n🎉 Setup complete!');
    console.log(`📧 Email: jagadeshvarma07@gmail.com`);
    console.log(`🏷️  Role: ${user.rolePrimary}`);
    console.log(`📊 Status: ${user.status}`);
    console.log('\n🚀 You can now access the admin panel in your dashboard!');

  } catch (error) {
    console.error('❌ Error setting up admin:', error.message);
    
    if (error.code === 'P1001') {
      console.log('\n💡 Database connection failed. Make sure your database is running and DATABASE_URL is correct.');
    } else if (error.code?.startsWith('P2')) {
      console.log('\n💡 Database schema issue. You may need to run: npx prisma migrate dev');
    } else {
      console.log('\n💡 Full error:', error);
    }
  } finally {
    await prisma.$disconnect();
  }
}

setupAdmin().catch(console.error);
