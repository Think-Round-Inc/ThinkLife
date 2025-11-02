import { NextRequest, NextResponse } from "next/server"
import { prisma } from "../../../../lib/prisma"
import crypto from "crypto"
import { Resend } from "resend"

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email) {
      return NextResponse.json(
        { error: "Email is required" },
        { status: 400 }
      )
    }

    // Check if Resend API key is configured
    if (!process.env.RESEND_API_KEY || process.env.RESEND_API_KEY.trim() === "") {
      console.error("RESEND_API_KEY is not configured properly")
      return NextResponse.json(
        { error: "Email service is not configured. Please set up Resend API key or contact support." },
        { status: 500 }
      )
    }

    // Check if user exists
    const user = await prisma.user.findUnique({
      where: { email }
    })

    if (!user) {
      // Don't reveal if user exists or not for security
      return NextResponse.json(
        { message: "If an account with that email exists, we've sent a password reset link." },
        { status: 200 }
      )
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(32).toString("hex")
    const resetTokenExpiry = new Date(Date.now() + 3600000) // 1 hour from now

    // Save reset token to database
    await prisma.user.update({
      where: { email },
      data: {
        resetToken,
        resetTokenExpiry
      }
    })

    // Initialize Resend with API key
    const resend = new Resend(process.env.RESEND_API_KEY)

    // Email content
    const resetUrl = `${process.env.NEXTAUTH_URL}/auth/reset-password?token=${resetToken}`

    // Send email using Resend
    await resend.emails.send({
      from: 'ThinkLife <noreply@thinklife.com>',
      to: [email],
      subject: 'Reset Your ThinkLife Password',
      html: `
        <div style="max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif;">
          <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
            <h1 style="color: white; margin: 0; font-size: 28px;">ThinkLife</h1>
            <p style="color: #e0e7ff; margin: 10px 0 0 0;">Password Reset Request</p>
          </div>

          <div style="background: white; padding: 40px; border-radius: 0 0 10px 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="color: #374151; margin-bottom: 20px;">Reset Your Password</h2>

            <p style="color: #6b7280; line-height: 1.6; margin-bottom: 30px;">
              We received a request to reset your password for your ThinkLife account.
              Click the button below to create a new password:
            </p>

            <div style="text-align: center; margin: 30px 0;">
              <a href="${resetUrl}"
                 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px 30px;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: bold;
                        display: inline-block;">
                Reset Password
              </a>
            </div>

            <p style="color: #9ca3af; font-size: 14px; line-height: 1.5;">
              This link will expire in 1 hour for security reasons. If you didn't request this password reset,
              you can safely ignore this email.
            </p>

            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">

            <p style="color: #9ca3af; font-size: 12px; text-align: center;">
              If the button doesn't work, copy and paste this link into your browser:<br>
              <span style="word-break: break-all;">${resetUrl}</span>
            </p>

            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; text-align: center;">
              <p style="color: #9ca3af; font-size: 12px; margin: 0;">
                This email was sent by ThinkLife, a product of Think Round, Inc.<br>
                2140 Bush Street, 1, San Francisco, CA 94115
              </p>
            </div>
          </div>
        </div>
      `,
    })

    return NextResponse.json(
      { message: "If an account with that email exists, we've sent a password reset link." },
      { status: 200 }
    )

  } catch (error) {
    console.error("Forgot password error:", error)
    return NextResponse.json(
      { error: "An error occurred. Please try again." },
      { status: 500 }
    )
  }
}
