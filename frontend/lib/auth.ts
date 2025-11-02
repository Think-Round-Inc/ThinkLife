import { NextAuthOptions } from "next-auth"
import { randomUUID } from "crypto"
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import GoogleProvider from "next-auth/providers/google"
import CredentialsProvider from "next-auth/providers/credentials"
import bcrypt from "bcryptjs"
import { prisma } from "./prisma"

declare module "next-auth" {
  interface Session {
    user: {
      id: string
      name?: string | null
      email?: string | null
      image?: string | null
    }
  }
}

export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  adapter: PrismaAdapter(prisma),
  providers: [
    // Only include OAuth providers if environment variables are present
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET ? [
      GoogleProvider({
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      })
    ] : []),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // Guest login fallback
        if (credentials.email === "guest@guest.com" && credentials.password === "ThinxLife") {
          return {
            id: "guest-user",
            email: "guest@thinklife.com",
            name: "Guest User",
          }
        }

        try {
          const user = await prisma.user.findUnique({
            where: {
              email: credentials.email
            }
          })

          if (!user || !user.password) {
            return null
          }

          const isPasswordValid = await bcrypt.compare(
            credentials.password,
            user.password
          )

          if (!isPasswordValid) {
            return null
          }

          return {
            id: user.id,
            email: user.email,
            name: user.name,
          }
        } catch (error) {
          console.error("Database error during authentication:", error)
          
          // If database fails and guest credentials are provided, allow guest login
          if (credentials.email === "guest@guest.com" && credentials.password === "ThinxLife") {
            return {
              id: "guest-user",
              email: "guest@thinklife.com",
              name: "Guest User (DB Fallback)",
            }
          }
          
          return null
        }
      }
    })
  ],
  session: {
    strategy: "jwt"
  },
  pages: {
    signIn: "/auth/signin",
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        session.user.id = token.id as string
      }
      return session
    },
  },
  events: {
    async createUser({ user }) {
      try {
        // Create a minimal User on first OAuth signup
        await prisma.user.create({
          data: {
            id: randomUUID(),
            email: user.email!,
            name: user.name,
            image: user.image,
            updatedAt: new Date(),
          },
        })
      } catch (error) {
        // Ignore if profile already exists or table is not ready
        console.warn('[auth] createUser profile creation skipped:', (error as Error)?.message)
      }
    },
  },
}
