"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Menu, X, User, LogOut, Activity, Heart, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { createClientComponentClient } from "@supabase/auth-helpers-nextjs";

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [authLoading, setAuthLoading] = useState(true);
  const [currentUser, setCurrentUser] = useState<{ name?: string | null; email?: string | null; image?: string | null; rolePrimary?: string } | null>(null);
  const supabase = createClientComponentClient();
  const pathname = usePathname();

  // Check if user is actively using features
  const isUsingAIAwareness = pathname?.startsWith('/inside-our-ai');
  const isUsingHealingRooms = pathname?.startsWith('/healing-rooms');
  const isUsingExteriorSpaces = pathname?.startsWith('/exterior-spaces');
  const isHome = pathname === '/';

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Supabase auth state
  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const { data } = await supabase.auth.getUser();
        if (!mounted) return;
        if (!data.user?.email) {
          setCurrentUser(null);
          return;
        }
        // Prefer RBAC profile data
        const rbacRes = await fetch('/api/rbac/profile', { cache: 'no-store' });
        if (rbacRes.ok) {
          const j = await rbacRes.json();
          setCurrentUser({ 
            name: j.user?.name || null, 
            email: j.user?.email || data.user.email, 
            image: j.user?.image || null,
            rolePrimary: j.user?.rolePrimary || 'MEMBER'
          });
        } else {
          // Fallback to old API
          const res = await fetch('/api/profile', { cache: 'no-store' });
          if (res.ok) {
            const j = await res.json();
            setCurrentUser({ 
              name: j.user?.name || null, 
              email: j.user?.email || data.user.email, 
              image: null,
              rolePrimary: 'MEMBER'
            });
          } else {
            setCurrentUser({
              name: (data.user.user_metadata as any)?.name || (data.user.user_metadata as any)?.full_name || null,
              email: data.user.email,
              image: (data.user.user_metadata as any)?.avatar_url || null,
              rolePrimary: 'MEMBER'
            });
          }
        }
      } finally {
        if (mounted) setAuthLoading(false);
      }
    };
    load();

    const { data: sub } = supabase.auth.onAuthStateChange((_event) => {
      load();
    });
    return () => {
      mounted = false;
      sub.subscription.unsubscribe();
    };
  }, [supabase]);

  const getInitials = (name: string | null | undefined) => {
    if (!name) return "U";
    return name.split(" ").map(n => n[0]).join("").toUpperCase();
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 transition-all duration-300">
      {/* Clean White Navigation Bar */}
      <nav className={`py-2 px-6 md:px-12 lg:px-24 transition-all duration-300 ${
        isScrolled
          ? 'bg-white/95 backdrop-blur-xl border-b border-gray-200/60 shadow-sm'
          : 'bg-white/90 backdrop-blur-md border-b border-gray-100/40'
      }`}>
        <div className="flex items-center justify-between">
          {/* Logo - Left */}
          <Link href="/" className="flex items-center group">
            <span className="text-xl font-bold tracking-wider text-black">
              ThinkLife
              <sup className="text-xs font-normal text-gray-700 ml-2 tracking-normal flex items-center">
                from <img src="/tr_logo.png" alt="Think Round" className="w-3 h-3 ml-1" /> Think Round, Inc
              </sup>
            </span>
          </Link>

          {/* Desktop Navigation - Center */}
          <div className="hidden md:flex items-center space-x-6">
            <Link 
              href="/inside-our-ai" 
              className={`relative text-gray-600 hover:text-black text-sm font-medium transition-colors duration-300 py-2 group ${
                isUsingAIAwareness ? 'text-black' : ''
              }`}
            >
              Inside our AI
              <div className={`absolute bottom-0 left-0 h-0.5 bg-black transition-all duration-300 ${
                isUsingAIAwareness ? 'w-full' : 'w-0 group-hover:w-full'
              }`}></div>
            </Link>
            <Link 
              href="/healing-rooms" 
              className={`relative text-gray-600 hover:text-black text-sm font-medium transition-colors duration-300 py-2 group ${
                isUsingHealingRooms ? 'text-black' : ''
              }`}
            >
              Healing Rooms
              <div className={`absolute bottom-0 left-0 h-0.5 bg-black transition-all duration-300 ${
                isUsingHealingRooms ? 'w-full' : 'w-0 group-hover:w-full'
              }`}></div>
            </Link>
            <Link 
              href="/exterior-spaces" 
              className={`relative text-gray-600 hover:text-black text-sm font-medium transition-colors duration-300 py-2 group ${
                isUsingExteriorSpaces ? 'text-black' : ''
              }`}
            >
              Exterior Spaces
              <div className={`absolute bottom-0 left-0 h-0.5 bg-black transition-all duration-300 ${
                isUsingExteriorSpaces ? 'w-full' : 'w-0 group-hover:w-full'
              }`}></div>
            </Link>
            <Link
              href="https://www.thinkround.org/aboutus"
              target="_blank"
              rel="noopener noreferrer"
              className="relative text-gray-600 hover:text-black font-medium transition-colors duration-300 text-sm py-2 group"
            >
              About
              <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-black transition-all duration-300 group-hover:w-full"></div>
            </Link>
            <Link
              href="https://thinkround.shop/"
              target="_blank"
              rel="noopener noreferrer"
              className="relative text-gray-600 hover:text-black font-medium transition-colors duration-300 text-sm py-2 group"
            >
              Shop
              <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-black transition-all duration-300 group-hover:w-full"></div>
            </Link>
            <Link
              href="https://www.thinkround.org/donate"
              target="_blank"
              rel="noopener noreferrer"
              className="relative text-gray-600 hover:text-black font-medium transition-colors duration-300 text-sm py-2 group"
            >
              Donate
              <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-black transition-all duration-300 group-hover:w-full"></div>
            </Link>
          </div>

          {/* Authentication Section - Right */}
          <div className="flex items-center space-x-3">
            {authLoading ? (
              <div className="w-8 h-8 bg-gray-200 rounded-full animate-pulse"></div>
            ) : currentUser ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="flex items-center space-x-2 hover:bg-gray-50 rounded-lg p-2 transition-all duration-300">
                    <Avatar className="w-7 h-7 border-2 border-gray-200">
                      <AvatarImage src={currentUser.image || undefined as any} />
                      <AvatarFallback className="bg-black text-white text-xs">
                        {getInitials(currentUser?.name)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="hidden md:block text-left">
                      <div className="text-sm font-medium text-gray-800">
                        {currentUser?.name?.split(' ')[0] || 'User'}
                      </div>
                    </div>
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-52 bg-white/95 backdrop-blur-sm border-gray-200 shadow-lg">
                  <div className="px-3 py-2">
                    <p className="text-sm font-medium text-gray-800">
                      {currentUser?.name || 'User'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {currentUser?.email}
                    </p>
                  </div>
                  <DropdownMenuSeparator className="bg-gray-200" />
                  <DropdownMenuItem asChild>
                    <Link href="/dashboard" className="flex items-center w-full px-3 py-2 text-sm hover:bg-gray-50">
                      <User className="w-4 h-4 mr-2" />
                      Dashboard
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/chat" className="flex items-center w-full px-3 py-2 text-sm hover:bg-gray-50">
                      <Activity className="w-4 h-4 mr-2" />
                      AI Assistant
                    </Link>
                  </DropdownMenuItem>
                  {currentUser?.rolePrimary === "ADMIN" && (
                    <DropdownMenuItem asChild>
                      <Link href="/admin" className="flex items-center w-full px-3 py-2 text-sm hover:bg-gray-50">
                        <Shield className="w-4 h-4 mr-2" />
                        Admin Dashboard
                      </Link>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuSeparator className="bg-gray-200" />
                  <DropdownMenuItem
                    onClick={async () => { await supabase.auth.signOut(); window.location.href = '/'; }}
                    className="flex items-center w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50"
                  >
                    <LogOut className="w-4 h-4 mr-2" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <div className="flex items-center space-x-2">
                {/* Combined Sign In / Sign Up Button */}
                <Link href="/auth/signin">
                  <Button className="bg-black hover:bg-gray-800 text-white px-6 py-2 rounded-full text-sm font-medium shadow-lg hover:shadow-black/25 transition-all duration-300 transform hover:scale-105">
                    Get Started
                  </Button>
                </Link>
              </div>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-50 transition-colors duration-300"
            >
              {isMenuOpen ? (
                <X className="w-5 h-5 text-gray-600" />
              ) : (
                <Menu className="w-5 h-5 text-gray-600" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden mt-3 py-3 bg-white/90 backdrop-blur-sm rounded-lg border border-gray-100 shadow-lg">
            <div className="flex flex-col space-y-2 px-4">
              <Link
                href="/inside-our-ai"
                className={`py-2 transition-colors duration-300 flex items-center ${
                  isUsingAIAwareness ? 'text-black font-medium' : 'text-gray-600 hover:text-black'
                }`}
                onClick={() => setIsMenuOpen(false)}
              >
                Inside our AI
                {isUsingAIAwareness && (
                  <div className="ml-2 w-1 h-4 bg-black rounded-full"></div>
                )}
              </Link>
              <Link
                href="/healing-rooms"
                className={`py-2 transition-colors duration-300 flex items-center ${
                  isUsingHealingRooms ? 'text-black font-medium' : 'text-gray-600 hover:text-black'
                }`}
                onClick={() => setIsMenuOpen(false)}
              >
                Healing Rooms
                {isUsingHealingRooms && (
                  <div className="ml-2 w-1 h-4 bg-black rounded-full"></div>
                )}
              </Link>
              <Link
                href="/exterior-spaces"
                className={`py-2 transition-colors duration-300 flex items-center ${
                  isUsingExteriorSpaces ? 'text-black font-medium' : 'text-gray-600 hover:text-black'
                }`}
                onClick={() => setIsMenuOpen(false)}
              >
                Exterior Spaces
                {isUsingExteriorSpaces && (
                  <div className="ml-2 w-1 h-4 bg-black rounded-full"></div>
                )}
              </Link>
              <Link
                href="https://www.thinkround.org/aboutus"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-600 hover:text-black py-2 transition-colors duration-300"
                onClick={() => setIsMenuOpen(false)}
              >
                About
              </Link>
              <Link
                href="https://thinkround.shop/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-600 hover:text-black py-2 transition-colors duration-300"
                onClick={() => setIsMenuOpen(false)}
              >
                Shop
              </Link>
              <Link
                href="https://www.thinkround.org/donate"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-600 hover:text-black py-2 transition-colors duration-300 flex items-center"
                onClick={() => setIsMenuOpen(false)}
              >
                <Heart className="w-4 h-4 mr-2" />
                Donate
              </Link>
              
              {!currentUser && (
                <div className="pt-3 border-t border-gray-100 space-y-2">
                  <Link href="/auth/signin" onClick={() => setIsMenuOpen(false)}>
                    <Button className="w-full bg-black hover:bg-gray-800 text-white rounded-full">
                      Get Started
                    </Button>
                  </Link>
                </div>
              )}
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}
