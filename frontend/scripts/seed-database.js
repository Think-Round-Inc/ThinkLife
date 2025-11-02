const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Seeding database...');

  // Create sample users
  const user1 = await prisma.user.create({
    data: {
      name: 'John Doe',
      email: 'john@example.com',
      firstName: 'John',
      lastName: 'Doe',
      bio: 'A passionate developer and wellness enthusiast.',
      phone: '+1-555-0123',
      location: 'San Francisco, CA',
      website: 'https://johndoe.com',
      theme: 'light',
      notifications: true,
      newsletter: true,
    },
  });

  const user2 = await prisma.user.create({
    data: {
      name: 'Jane Smith',
      email: 'jane@example.com',
      firstName: 'Jane',
      lastName: 'Smith',
      bio: 'Mental health advocate and yoga instructor.',
      phone: '+1-555-0456',
      location: 'New York, NY',
      website: 'https://janesmith.yoga',
      theme: 'dark',
      notifications: true,
      newsletter: false,
    },
  });

  // Create profiles for users
  const profile1 = await prisma.profile.create({
    data: {
      userId: user1.id,
      occupation: 'Software Engineer',
      company: 'Tech Corp',
      interests: ['Technology', 'Meditation', 'Hiking', 'Reading'],
      skills: ['JavaScript', 'React', 'Node.js', 'Mindfulness'],
      socialLinks: {
        twitter: 'https://twitter.com/johndoe',
        linkedin: 'https://linkedin.com/in/johndoe',
        github: 'https://github.com/johndoe'
      },
      privacy: 'public',
    },
  });

  const profile2 = await prisma.profile.create({
    data: {
      userId: user2.id,
      occupation: 'Yoga Instructor',
      company: 'Mindful Wellness Center',
      interests: ['Yoga', 'Meditation', 'Wellness', 'Nature'],
      skills: ['Hatha Yoga', 'Vinyasa', 'Meditation', 'Breathwork'],
      socialLinks: {
        instagram: 'https://instagram.com/janesmith_yoga',
        website: 'https://janesmith.yoga'
      },
      privacy: 'public',
    },
  });

  // Create sample experiences
  const experience1 = await prisma.experience.create({
    data: {
      userId: user1.id,
      name: 'John Doe',
      email: 'john@example.com',
      title: 'Life-Changing Meditation Journey',
      content: 'ThinkLife has completely transformed my approach to mental wellness. The AI-guided meditation sessions helped me develop a consistent practice that fits my busy schedule. I\'ve noticed significant improvements in my stress levels and overall well-being.',
      rating: 5,
      isPublic: true,
      isApproved: true,
    },
  });

  const experience2 = await prisma.experience.create({
    data: {
      userId: user2.id,
      name: 'Jane Smith',
      email: 'jane@example.com',
      title: 'Professional Growth Through Mindfulness',
      content: 'As a yoga instructor, I was skeptical about AI-assisted wellness tools. However, ThinkLife\'s personalized approach and deep understanding of mindfulness principles impressed me. It\'s become an invaluable resource for both my personal practice and teaching.',
      rating: 5,
      isPublic: true,
      isApproved: true,
    },
  });

  const experience3 = await prisma.experience.create({
    data: {
      name: 'Anonymous User',
      title: 'Overcoming Anxiety',
      content: 'I was dealing with severe anxiety and found traditional therapy wasn\'t enough. The combination of AI companionship and therapeutic techniques on ThinkLife provided the extra support I needed. The 24/7 availability was crucial during my difficult moments.',
      rating: 4,
      isPublic: true,
      isApproved: true,
    },
  });

  const experience4 = await prisma.experience.create({
    data: {
      name: 'Sarah Johnson',
      email: 'sarah@example.com',
      title: 'Healing Rooms Experience',
      content: 'The virtual healing rooms created such a peaceful environment for my meditation practice. The ambient sounds and guided visualizations helped me achieve deeper states of relaxation than I ever thought possible.',
      rating: 5,
      isPublic: true,
      isApproved: true,
    },
  });

  const experience5 = await prisma.experience.create({
    data: {
      name: 'Mike Chen',
      email: 'mike@example.com',
      title: 'Stress Management Success',
      content: 'Working in a high-pressure environment, I needed effective stress management tools. ThinkLife\'s personalized recommendations and breathing exercises have made a real difference in my daily life. Highly recommended!',
      rating: 4,
      isPublic: true,
      isApproved: false, // This one is pending approval
    },
  });

  console.log('✅ Database seeded successfully!');
  console.log(`📊 Created:`);
  console.log(`   - ${2} users`);
  console.log(`   - ${2} profiles`);
  console.log(`   - ${5} experiences`);
  console.log('');
  console.log('🎉 You can now view the data in Prisma Studio at http://localhost:5555');
}

main()
  .catch((e) => {
    console.error('❌ Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  }); 