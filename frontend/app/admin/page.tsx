import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { createServerComponentClient } from "@supabase/auth-helpers-nextjs";
import Link from "next/link";

function isAdminEmail(email?: string | null): boolean {
  if (!email) return false;
  const raw = process.env.ADMIN_EMAILS || process.env.NEXT_PUBLIC_ADMIN_EMAILS || "";
  const list = raw
    .split(/[\s,]+/)
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean);
  return list.includes(email.toLowerCase());
}

export default async function AdminPage() {
  const supabase = createServerComponentClient({ cookies });
  const { data } = await supabase.auth.getUser();

  if (!data.user) {
    redirect("/auth/signin");
  }

  // Prefer RBAC profile role; fall back to admin email list
  let isAdmin = isAdminEmail(data.user.email);
  try {
    const res = await fetch("/api/rbac/profile", { cache: "no-store" });
    if (res.ok) {
      const j = await res.json();
      if (j?.user?.rolePrimary === "ADMIN") {
        isAdmin = true;
      }
    }
  } catch {}

  if (!isAdmin) {
    redirect("/dashboard");
  }

  return (
    <main className="min-h-screen bg-gray-50 pt-16">
      <div className="max-w-5xl mx-auto p-6">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
          <Link href="/dashboard" className="text-sm text-blue-600 hover:text-blue-700">Back to Dashboard</Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <section className="p-6 bg-white border rounded-lg shadow-sm">
            <h2 className="text-lg font-semibold mb-2">Access Requests</h2>
            <p className="text-sm text-gray-600 mb-4">Review and manage user access requests.</p>
            <Link href="/dashboard" className="text-blue-600 hover:text-blue-700 text-sm">Open in Dashboard »</Link>
          </section>

          <section className="p-6 bg-white border rounded-lg shadow-sm">
            <h2 className="text-lg font-semibold mb-2">User Management</h2>
            <p className="text-sm text-gray-600 mb-4">Manage user roles and permissions.</p>
            <span className="text-gray-500 text-sm">Coming soon</span>
          </section>
        </div>
      </div>
    </main>
  );
}


