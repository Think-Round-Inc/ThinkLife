import { NextRequest, NextResponse } from "next/server"
import { prisma } from "../../../lib/prisma"
import { z } from "zod"
import bcrypt from "bcryptjs"
import { randomUUID } from "crypto"
import { cookies } from "next/headers"
import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs"

const profileUpdateSchema = z.object({
  name: z.string().optional(),
  age: z.number().int().min(1).max(120).optional(),
  password: z.string().min(6).optional(),
  bio: z.string().max(500).optional(),
  location: z.string().max(100).optional(),
  website: z.string().url().optional().or(z.literal("")),
  github: z.string().max(100).optional(),
  discord: z.string().max(100).optional(),
  phone: z.string().max(20).optional(),
  theme: z.enum(["light", "dark"]).optional(),
  notifications: z.boolean().optional(),
  newsletter: z.boolean().optional(),
})

// GET /api/profile - Get current user's profile
export async function GET(request: NextRequest) {
  try {
    const supabase = createRouteHandlerClient({ cookies })
    const { data } = await supabase.auth.getUser()

    if (!data.user?.email) {
      console.error("Profile API: No session or email found")
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    // Note: Prisma auto-connects on first query, no need to manually connect

    console.log("Profile API: Looking for user with email:", data.user.email)

    // Handle guest users who don't exist in database
    if (data.user.email === "guest@thinklife.com") {
      console.log("Profile API: Returning guest user profile")
      const guestUser = {
        id: "guest-user",
        name: "Guest User",
        email: "guest@thinklife.com",
        firstName: "Guest",
        lastName: "User",
        bio: "Welcome to ThinkLife! This is a guest account for exploring our platform.",
        phone: null,
        dateOfBirth: null,
        location: null,
        website: null,
        theme: "light",
        notifications: true,
        newsletter: false,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
      return NextResponse.json({ user: guestUser })
    }

    // Check if DATABASE_URL is available
    if (!process.env.DATABASE_URL) {
      console.log("No DATABASE_URL found, using Supabase fallback data");
      const fallbackUser = {
        id: data.user.id,
        name: data.user.user_metadata?.name || data.user.user_metadata?.full_name || null,
        email: data.user.email,
        firstName: null,
        lastName: null,
        bio: null,
        phone: null,
        dateOfBirth: null,
        location: null,
        website: null,
        github: null,
        discord: null,
        theme: "light",
        notifications: true,
        newsletter: false,
        createdAt: data.user.created_at,
        updatedAt: data.user.updated_at,
        rolePrimary: "ADMIN", // Set to ADMIN for testing
        status: "ACTIVE",
        TeamMemberships: [],
        UserRoles: [],
      };
      return NextResponse.json({ user: fallbackUser });
    }

    // Query user from database
    let user = await prisma.user.findUnique({
      where: { email: data.user.email! }
    })

    if (!user) {
      // First Google login: create a minimal local user seeded with Google name
      const googleName = ((data.user.user_metadata as any)?.name 
        || (data.user.user_metadata as any)?.full_name 
        || null) as string | null

      const created = await prisma.user.create({
        data: {
          id: randomUUID(),
          email: data.user.email!,
          name: googleName,
          updatedAt: new Date(),
        }
      })
      user = created
    }

    console.log("Profile API: Successfully found user:", user.id)
    return NextResponse.json({ user })
  } catch (error: any) {
    console.error("Profile fetch error:", error)
    console.error("Error details:", {
      message: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : 'No stack trace',
      name: error instanceof Error ? error.name : 'Unknown error type',
      code: error?.code,
      clientVersion: error?.clientVersion
    })

    // Handle database errors
    if (error?.code?.startsWith('P') || error?.clientVersion) {
      return NextResponse.json(
        { 
          error: "Database error", 
          details: "We're having trouble accessing your data. Please try again.",
        },
        { status: 503 }
      )
    }

    // Generic error fallback
    return NextResponse.json(
      { 
        error: "Internal server error", 
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

// PUT /api/profile - Update current user's profile
export async function PUT(request: NextRequest) {
  try {
    const supabase = createRouteHandlerClient({ cookies })
    const { data } = await supabase.auth.getUser()

    if (!data.user?.email) {
      console.error("Profile Update API: No session or email found")
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    const body = await request.json()
    const validatedData = profileUpdateSchema.parse(body)

    console.log("Profile Update API: User email:", data.user.email)

    // Handle guest users who can't update database
    if (data.user.email === "guest@thinklife.com") {
      console.log("Profile Update API: Guest user attempted to update profile")
      return NextResponse.json(
        { error: "Guest users cannot update their profile. Please create a full account to save changes." },
        { status: 403 }
      )
    }

    // Check if DATABASE_URL is available
    if (!process.env.DATABASE_URL) {
      console.log("No DATABASE_URL found, profile updates not available in fallback mode");
      return NextResponse.json(
        { error: "Profile updates require database connection. Please configure DATABASE_URL." },
        { status: 503 }
      )
    }

    // Note: Prisma auto-connects on first query, no need to manually connect

    // Update user in database
    // Prepare updates
    const updates: any = { updatedAt: new Date() }
    if (validatedData.name !== undefined) updates.name = validatedData.name
    if (validatedData.age !== undefined) {
      // Convert age to an approximate dateOfBirth (Jan 1 of birth year)
      const birthYear = new Date().getFullYear() - validatedData.age
      updates.dateOfBirth = new Date(birthYear, 0, 1)
    }
    if (validatedData.password !== undefined) {
      updates.password = await bcrypt.hash(validatedData.password, 12)
    }
    if (validatedData.bio !== undefined) updates.bio = validatedData.bio
    if (validatedData.location !== undefined) updates.location = validatedData.location
    if (validatedData.website !== undefined) updates.website = validatedData.website
    if (validatedData.github !== undefined) updates.github = validatedData.github
    if (validatedData.discord !== undefined) updates.discord = validatedData.discord
    if (validatedData.phone !== undefined) updates.phone = validatedData.phone
    if (validatedData.theme !== undefined) updates.theme = validatedData.theme
    if (validatedData.notifications !== undefined) updates.notifications = validatedData.notifications
    if (validatedData.newsletter !== undefined) updates.newsletter = validatedData.newsletter

    const updatedUser = await prisma.user.update({
      where: { email: data.user.email! },
      data: updates
    })

    // Attempt to sync Supabase user metadata/password
    try {
      const supabaseUpdates: any = {}
      if (validatedData.name !== undefined) {
        supabaseUpdates.data = { ...(supabaseUpdates.data || {}), name: validatedData.name }
      }
      if (validatedData.password !== undefined) {
        supabaseUpdates.password = validatedData.password
      }
      if (Object.keys(supabaseUpdates).length > 0) {
        const supabase = createRouteHandlerClient({ cookies })
        await supabase.auth.updateUser(supabaseUpdates)
      }
    } catch (e) {
      console.warn('[profile] Supabase sync failed:', (e as Error)?.message)
    }

    console.log("Profile Update API: Successfully updated user:", updatedUser.id)
    return NextResponse.json({
      message: "Profile updated successfully",
      user: updatedUser
    })
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: "Invalid data", details: error.issues },
        { status: 400 }
      )
    }

    console.error("Profile update error:", error)
    console.error("Error details:", {
      message: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : 'No stack trace',
      name: error instanceof Error ? error.name : 'Unknown error type',
      code: error?.code,
      clientVersion: error?.clientVersion
    })

    // Handle database errors
    if (error?.code?.startsWith('P') || error?.clientVersion) {
      return NextResponse.json(
        { 
          error: "Database error", 
          details: "We're having trouble accessing your data. Please try again.",
        },
        { status: 503 }
      )
    }

    // Generic error fallback
    return NextResponse.json(
      { 
        error: "Internal server error", 
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
