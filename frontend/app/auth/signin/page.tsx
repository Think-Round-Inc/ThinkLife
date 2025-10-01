"use client"

import Link from "next/link"
import SupabaseGoogleButton from "@/components/auth/supabase-google-button"

export default function SignInPage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 via-purple-100 to-blue-50 p-6">
      <div className="w-full max-w-md rounded-2xl bg-white/90 backdrop-blur-md shadow-xl border border-purple-200/50 p-8 space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-800 to-purple-700 bg-clip-text text-transparent">
            Sign in to ThinkxLife
          </h1>
          <p className="text-slate-600">Welcome back. Choose a sign-in method to continue.</p>
        </div>

        <div className="space-y-4">
          <SupabaseGoogleButton className="w-full" />
        </div>

        <div className="text-center text-sm text-slate-600">
          <Link href="/auth/forgot-password" className="text-purple-600 hover:text-purple-700 font-medium">
            Forgot password?
          </Link>
        </div>
      </div>
    </main>
  )
}


