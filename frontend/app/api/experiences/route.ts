import { NextRequest, NextResponse } from "next/server"
// NextAuth removed
import { prisma } from "../../../lib/prisma"
import { z } from "zod"

const experienceSubmissionSchema = z.object({
  name: z.string().min(1, "Name is required").max(100, "Name must be under 100 characters"),
  email: z.string().email("Invalid email").optional().or(z.literal("")),
  title: z.string().max(200, "Title must be under 200 characters").optional().or(z.literal("")),
  content: z.string().min(10, "Experience must be at least 10 characters").max(2000, "Experience must be under 2000 characters"),
  rating: z.number().min(1).max(5).optional(),
})

// GET /api/experiences - Get approved public experiences
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '10')
    const offset = parseInt(searchParams.get('offset') || '0')

    const experiences = await (prisma as any).experience.findMany({
      where: {
        isPublic: true,
        isApproved: true,
      },
      select: {
        id: true,
        name: true,
        title: true,
        content: true,
        rating: true,
        createdAt: true,
      },
      orderBy: {
        createdAt: 'desc'
      },
      take: Math.min(limit, 50), // Max 50 at a time
      skip: offset,
    })

    const totalCount = await (prisma as any).experience.count({
      where: {
        isPublic: true,
        isApproved: true,
      }
    })

    return NextResponse.json({
      experiences,
      totalCount,
      hasMore: offset + experiences.length < totalCount
    })

  } catch (error: any) {
    console.error("Experience fetch error:", error)
    
    // Handle case where Experience table doesn't exist yet or database connection issues
    if (error.code === 'P2021' || 
        error.message?.includes('does not exist') ||
        error.code === 'P1001' || // Cannot reach database server
        error.code === 'P2002' || // Unique constraint violation (unlikely here but safe to handle)
        error.name === 'PrismaClientInitializationError') {
      // Return empty array gracefully instead of error
      return NextResponse.json({
        experiences: [],
        totalCount: 0,
        hasMore: false
      })
    }
    
    // For other errors, still return empty array to prevent frontend crashes
    console.warn("Experience fetch error (returning empty):", error.message || error)
    return NextResponse.json({
      experiences: [],
      totalCount: 0,
      hasMore: false
    })
  }
}

// POST /api/experiences - Submit a new experience
export async function POST(request: NextRequest) {
  try {
    const session = null
    const body = await request.json()

    // Validate the submission
    const validatedData = experienceSubmissionSchema.parse(body)

    // Create the experience
    const experience = await (prisma as any).experience.create({
      data: {
        userId: null,
        name: validatedData.name,
        email: validatedData.email || null,
        title: validatedData.title || null,
        content: validatedData.content,
        rating: validatedData.rating || null,
        isPublic: true,
        isApproved: false, // Requires approval for moderation
        updatedAt: new Date(),
      },
      select: {
        id: true,
        name: true,
        title: true,
        content: true,
        rating: true,
        createdAt: true,
      }
    })

    return NextResponse.json({
      message: "Experience submitted successfully! It will be reviewed before being published.",
      experience
    }, { status: 201 })

  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { 
          error: "Invalid submission", 
          details: error.issues.map(err => ({
            field: err.path.join('.'),
            message: err.message
          }))
        },
        { status: 400 }
      )
    }

    console.error("Experience submission error:", error)
    
    // Handle case where Experience table doesn't exist yet
    if (error.code === 'P2021' || error.message?.includes('does not exist')) {
      return NextResponse.json(
        { error: "Experience feature is temporarily unavailable. Please try again later." },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { error: "Failed to submit experience" },
      { status: 500 }
    )
  }
} 