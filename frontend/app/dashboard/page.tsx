"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import {
  User,
  Users,
  Settings,
  Bell,
  Shield,
  Key,
  Mail,
  Phone,
  MapPin,
  Globe,
  Github,
  MessageCircle,
  Calendar,
  Activity,
  FileText,
  HelpCircle,
  Edit3,
  Save,
  X,
  Loader2,
  CheckCircle,
  AlertTriangle,
  Crown,
  Star,
  Eye,
  EyeOff,
  Plus,
  Clock,
  XCircle,
} from "lucide-react";
import { createClientComponentClient } from "@supabase/auth-helpers-nextjs";

type UserRole = "MEMBER" | "INTERN" | "TEAM_LEAD" | "ADMIN";
type UserStatus = "ACTIVE" | "SUSPENDED" | "ARCHIVED";

interface UserProfile {
  id: string;
  email: string;
  name?: string;
  image?: string;
  rolePrimary: UserRole;
  status: UserStatus;
  createdAt: string;
  updatedAt: string;
  dateOfBirth?: string;
  location?: string;
  website?: string;
  bio?: string;
  phone?: string;
  github?: string;
  discord?: string;
  theme?: string;
  notifications: boolean;
  newsletter: boolean;
  TeamMemberships?: Array<{
    teamId: string;
    Team: {
      id: string;
      name: string;
      description?: string;
      leadId?: string;
    };
  }>;
  UserRoles?: Array<{
    Role: {
      code: string;
      name: string;
      description?: string;
    };
  }>;
}

interface AccessRequest {
  id: string;
  type: string;
  target: string;
  reason?: string;
  status: string;
  createdAt: string;
}

const roleColors: Record<UserRole, string> = {
  MEMBER: "bg-blue-100 text-blue-800 border-blue-200",
  INTERN: "bg-green-100 text-green-800 border-green-200",
  TEAM_LEAD: "bg-purple-100 text-purple-800 border-purple-200",
  ADMIN: "bg-red-100 text-red-800 border-red-200",
};

const roleIcons: Record<UserRole, React.ComponentType<any>> = {
  MEMBER: User,
  INTERN: Star,
  TEAM_LEAD: Crown,
  ADMIN: Shield,
};

const statusColors: Record<UserStatus, string> = {
  ACTIVE: "bg-green-100 text-green-800 border-green-200",
  SUSPENDED: "bg-yellow-100 text-yellow-800 border-yellow-200",
  ARCHIVED: "bg-gray-100 text-gray-800 border-gray-200",
};

const sidebarItems = [
  { id: "profile", label: "Profile", icon: User },
  { id: "teams", label: "Teams", icon: Users },
  { id: "requests", label: "My Requests", icon: Clock },
  { id: "admin", label: "Admin Panel", icon: Shield, adminOnly: true },
  { id: "security", label: "Security", icon: Shield },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "settings", label: "Settings", icon: Settings },
  { id: "help", label: "Help & Support", icon: HelpCircle },
];

export default function ModernDashboard() {
  const supabase = createClientComponentClient();
  const router = useRouter();
  
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [accessRequests, setAccessRequests] = useState<AccessRequest[]>([]);
  const [allAccessRequests, setAllAccessRequests] = useState<AccessRequest[]>([]);
  const [activeSection, setActiveSection] = useState("profile");
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  
  // Modal states
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [show2FAModal, setShow2FAModal] = useState(false);
  const [showSessionsModal, setShowSessionsModal] = useState(false);
  const [showTeamModal, setShowTeamModal] = useState(false);

  // Form state
  const [formData, setFormData] = useState({
    name: "",
    bio: "",
    location: "",
    website: "",
    github: "",
    discord: "",
    phone: "",
    theme: "light",
    notifications: true,
    newsletter: false,
    newPassword: "",
    confirmPassword: "",
  });

  // Access request form
  const [newRequest, setNewRequest] = useState({
    type: "ROLE_CHANGE",
    target: "",
    reason: "",
  });

  useEffect(() => {
    fetchProfile();
    fetchAccessRequests();
    if (profile?.rolePrimary === "ADMIN") {
      fetchAllAccessRequests();
    }
  }, [profile?.rolePrimary]);

  const fetchProfile = async () => {
    try {
      const response = await fetch("/api/rbac/profile", { cache: "no-store" });
      if (response.ok) {
        const data = await response.json();
        setProfile(data.user);
        setFormData({
          name: data.user.name || "",
          bio: data.user.bio || "",
          location: data.user.location || "",
          website: data.user.website || "",
          github: data.user.github || "",
          discord: data.user.discord || "",
          phone: data.user.phone || "",
          theme: data.user.theme || "light",
          notifications: data.user.notifications ?? true,
          newsletter: data.user.newsletter ?? false,
          newPassword: "",
          confirmPassword: "",
        });
      } else {
        const errorData = await response.json();
        setError(errorData.error || "Failed to load profile");
      }
    } catch (error) {
      setError("Network error loading profile");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAccessRequests = async () => {
    try {
      // Use simple API directly for better reliability
      const response = await fetch("/api/access-requests");
      
      if (response.ok) {
        const data = await response.json();
        setAccessRequests(data.requests || []);
      }
    } catch (error) {
      console.error("Failed to load access requests:", error);
    }
  };

  const fetchAllAccessRequests = async () => {
    try {
      const response = await fetch("/api/access-requests/all");
      
      if (response.ok) {
        const data = await response.json();
        setAllAccessRequests(data.requests || []);
      }
    } catch (error) {
      console.error("Failed to load all access requests:", error);
    }
  };

  const handleRequestDecision = async (requestId: string, decision: "APPROVED" | "REJECTED") => {
    try {
      const response = await fetch(`/api/access-requests/${requestId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: decision }),
      });

      if (response.ok) {
        setSuccess(`Request ${decision.toLowerCase()} successfully!`);
        // Update the local state
        setAllAccessRequests(prev => 
          prev.map(req => 
            req.id === requestId ? { ...req, status: decision } : req
          )
        );
      } else {
        const data = await response.json();
        setError(data.error || `Failed to ${decision.toLowerCase()} request`);
      }
    } catch (error) {
      console.error("Request decision error:", error);
      setError("Network error. Please try again.");
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError("");
    setSuccess("");

    try {
      // Validate password confirmation if password is being changed
      if (formData.newPassword && formData.newPassword !== formData.confirmPassword) {
        setError("Passwords do not match");
        setIsSaving(false);
        return;
      }

      const { newPassword, confirmPassword, ...updateData } = formData;
      const finalUpdateData = newPassword ? { ...updateData, password: newPassword } : updateData;

      // Try RBAC API first
      let response = await fetch("/api/rbac/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalUpdateData),
      });

      // If RBAC API fails, try the old profile API
      if (response.status === 503) {
        response = await fetch("/api/profile", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(finalUpdateData),
        });
      }

      const data = await response.json();

      if (response.ok) {
        // Update the profile state with the new data
        setProfile(prev => ({ ...prev, ...data.user }));
        setSuccess("Profile updated successfully!");
        setIsEditing(false);
        setFormData(prev => ({ ...prev, newPassword: "", confirmPassword: "" }));
        
        // Refresh profile data
        await fetchProfile();
      } else {
        setError(data.error || "Failed to update profile");
      }
    } catch (error) {
      console.error("Profile update error:", error);
      setError("Network error. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleAccessRequest = async () => {
    try {
      // Validate form
      if (!newRequest.target.trim()) {
        setError("Please specify a target for your request");
        return;
      }

      // Use simple API directly for better reliability
      const response = await fetch("/api/access-requests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newRequest),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess("Access request submitted successfully!");
        // Add the new request to the local state immediately for better UX
        const newRequestWithId = {
          id: data.request?.id || `temp-${Date.now()}`,
          type: newRequest.type,
          target: newRequest.target,
          reason: newRequest.reason || "",
          status: "PENDING",
          createdAt: new Date().toISOString(),
        };
        setAccessRequests(prev => [newRequestWithId, ...prev]);
        setNewRequest({ type: "ROLE_CHANGE", target: "", reason: "" });
      } else {
        setError(data.error || "Failed to submit request");
      }
    } catch (error) {
      console.error("Access request error:", error);
      setError("Network error. Please try again.");
    }
  };



  const getInitials = (name?: string) => {
    if (!name) return "U";
    return name.split(" ").map(n => n[0]).join("").toUpperCase();
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="h-8 w-8 text-red-600 mx-auto mb-4" />
          <p className="text-gray-600">Failed to load profile</p>
          <Button onClick={() => router.push("/")} className="mt-4">
            Go Home
          </Button>
        </div>
      </div>
    );
  }

  const RoleIcon = roleIcons[profile.rolePrimary];

  const renderContent = () => {
    switch (activeSection) {
      case "profile":
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900">Profile Information</h2>
              <Button
                onClick={() => setIsEditing(!isEditing)}
                variant={isEditing ? "outline" : "default"}
                className={isEditing ? "border-red-300 text-red-700 hover:bg-red-50" : ""}
              >
                {isEditing ? (
                  <>
                    <X className="w-4 h-4 mr-2" />
                    Cancel
                  </>
                ) : (
                  <>
                    <Edit3 className="w-4 h-4 mr-2" />
                    Edit Profile
                  </>
                )}
              </Button>
            </div>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center space-x-4 mb-6">
                  <Avatar className="w-20 h-20">
                    <AvatarImage src={profile.image || undefined} />
                    <AvatarFallback className="bg-blue-600 text-white text-xl">
                      {getInitials(profile.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="text-lg font-semibold">{profile.name || "Unnamed User"}</h3>
                    <p className="text-sm text-gray-600">{profile.email}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge className={roleColors[profile.rolePrimary]}>
                        <RoleIcon className="w-3 h-3 mr-1" />
                        {profile.rolePrimary.replace("_", " ")}
                      </Badge>
                      <Badge className={statusColors[profile.status]}>
                        {profile.status}
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="name">Full Name</Label>
                      <Input
                        id="name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        value={profile.email}
                        disabled
                        className="mt-1 bg-gray-50"
                      />
                    </div>
                    <div>
                      <Label htmlFor="location">Location</Label>
                      <Input
                        id="location"
                        value={formData.location}
                        onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                        placeholder="City, Country"
                      />
                    </div>
                    <div>
                      <Label htmlFor="phone">Phone</Label>
                      <Input
                        id="phone"
                        value={formData.phone}
                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                        placeholder="+1 (555) 123-4567"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="website">Website</Label>
                      <Input
                        id="website"
                        value={formData.website}
                        onChange={(e) => setFormData({ ...formData, website: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                        placeholder="https://example.com"
                      />
                    </div>
                    <div>
                      <Label htmlFor="github">GitHub Username</Label>
                      <Input
                        id="github"
                        value={formData.github}
                        onChange={(e) => setFormData({ ...formData, github: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                        placeholder="username"
                      />
                    </div>
                    <div>
                      <Label htmlFor="discord">Discord Username</Label>
                      <Input
                        id="discord"
                        value={formData.discord}
                        onChange={(e) => setFormData({ ...formData, discord: e.target.value })}
                        disabled={!isEditing}
                        className="mt-1"
                        placeholder="username#1234"
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-6">
                  <Label htmlFor="bio">Bio</Label>
                  <Textarea
                    id="bio"
                    value={formData.bio}
                    onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
                    disabled={!isEditing}
                    className="mt-1"
                    placeholder="Tell us about yourself..."
                    rows={4}
                  />
                </div>

                {isEditing && (
                  <div className="mt-6 pt-6 border-t">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="newPassword">New Password</Label>
                        <Input
                          id="newPassword"
                          type="password"
                          value={formData.newPassword}
                          onChange={(e) => setFormData({ ...formData, newPassword: e.target.value })}
                          className="mt-1"
                          placeholder="Leave empty to keep current"
                        />
                      </div>
                      <div>
                        <Label htmlFor="confirmPassword">Confirm Password</Label>
                        <Input
                          id="confirmPassword"
                          type="password"
                          value={formData.confirmPassword}
                          onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                          className="mt-1"
                          placeholder="Confirm new password"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {isEditing && (
                  <div className="flex gap-3 mt-6 pt-6 border-t">
                    <Button
                      onClick={handleSave}
                      disabled={isSaving}
                      className="bg-green-600 hover:bg-green-700"
                    >
                      {isSaving ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="w-4 h-4 mr-2" />
                          Save Changes
                        </>
                      )}
                    </Button>
                    <Button
                      onClick={() => setIsEditing(false)}
                      variant="outline"
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        );

      case "teams":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Team Memberships</h2>
            
            {profile.TeamMemberships && profile.TeamMemberships.length > 0 ? (
              <div className="grid gap-4">
                {profile.TeamMemberships.map((membership) => (
                  <Card key={membership.teamId}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold">{membership.Team.name}</h3>
                          {membership.Team.description && (
                            <p className="text-gray-600 mt-1">{membership.Team.description}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {membership.Team.leadId === profile.id && (
                            <Badge className="bg-purple-100 text-purple-800 border-purple-200">
                              <Crown className="w-3 h-3 mr-1" />
                              Team Lead
                            </Badge>
                          )}
                          <Button variant="outline" size="sm">
                            <Eye className="w-4 h-4 mr-2" />
                            View Team
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No Team Memberships</h3>
                  <p className="text-gray-600 mb-4">You're not a member of any teams yet.</p>
                  <Dialog open={showTeamModal} onOpenChange={setShowTeamModal}>
                    <DialogTrigger asChild>
                      <Button className="bg-blue-600 hover:bg-blue-700">
                        <Plus className="w-4 h-4 mr-2" />
                        Request to Join Team
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Request to Join Team</DialogTitle>
                        <DialogDescription>
                          Submit a request to join an existing team.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="team-name">Team Name</Label>
                          <Select>
                            <SelectTrigger className="mt-1">
                              <SelectValue placeholder="Select a team" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="engineering">Engineering Team</SelectItem>
                              <SelectItem value="design">Design Team</SelectItem>
                              <SelectItem value="marketing">Marketing Team</SelectItem>
                              <SelectItem value="product">Product Team</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label htmlFor="join-reason">Why do you want to join this team?</Label>
                          <Textarea
                            id="join-reason"
                            placeholder="Explain your interest and relevant experience..."
                            className="mt-1"
                            rows={3}
                          />
                        </div>
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setShowTeamModal(false)}>
                          Cancel
                        </Button>
                        <Button onClick={() => {
                          setSuccess("Team join request submitted successfully!");
                          setShowTeamModal(false);
                        }}>
                          Submit Request
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </CardContent>
              </Card>
            )}
          </div>
        );

      case "requests":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">My Access Requests</h2>
            
            {/* New Request Form */}
            <Card>
              <CardHeader>
                <CardTitle>Submit New Request</CardTitle>
                <CardDescription>
                  Request role changes, team membership, or additional permissions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Request Type</Label>
                    <Select
                      value={newRequest.type}
                      onValueChange={(value) => setNewRequest({ ...newRequest, type: value })}
                    >
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ROLE_CHANGE">Role Change</SelectItem>
                        <SelectItem value="TEAM_JOIN">Join Team</SelectItem>
                        <SelectItem value="RESOURCE_ACCESS">Resource Access</SelectItem>
                        <SelectItem value="PERMISSION_REQUEST">Permission Request</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Target</Label>
                    <Input
                      value={newRequest.target}
                      onChange={(e) => setNewRequest({ ...newRequest, target: e.target.value })}
                      className="mt-1"
                      placeholder={
                        newRequest.type === "ROLE_CHANGE" ? "intern, team_lead, admin" :
                        newRequest.type === "TEAM_JOIN" ? "Team ID" :
                        "Resource or permission name"
                      }
                    />
                  </div>
                </div>
                <div>
                  <Label>Reason (optional)</Label>
                  <Textarea
                    value={newRequest.reason}
                    onChange={(e) => setNewRequest({ ...newRequest, reason: e.target.value })}
                    className="mt-1"
                    placeholder="Explain why you need this access..."
                    rows={3}
                  />
                </div>
                <Button
                  onClick={handleAccessRequest}
                  className="bg-blue-600 hover:bg-blue-700"
                  disabled={!newRequest.target}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Submit Request
                </Button>
              </CardContent>
            </Card>

            {/* Existing Requests */}
            <Card>
              <CardHeader>
                <CardTitle>Request History</CardTitle>
              </CardHeader>
              <CardContent>
                {accessRequests.length > 0 ? (
                  <div className="space-y-4">
                    {accessRequests.map((request) => (
                      <div
                        key={request.id}
                        className="p-4 border rounded-lg bg-white"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="outline">
                                {request.type.replace("_", " ")}
                              </Badge>
                              <Badge
                                className={
                                  request.status === "PENDING" ? "bg-yellow-100 text-yellow-800 border-yellow-200" :
                                  request.status === "APPROVED" ? "bg-green-100 text-green-800 border-green-200" :
                                  "bg-red-100 text-red-800 border-red-200"
                                }
                              >
                                {request.status === "PENDING" && <Clock className="w-3 h-3 mr-1" />}
                                {request.status === "APPROVED" && <CheckCircle className="w-3 h-3 mr-1" />}
                                {request.status === "REJECTED" && <XCircle className="w-3 h-3 mr-1" />}
                                {request.status}
                              </Badge>
                            </div>
                            <p className="font-medium">Target: {request.target}</p>
                            {request.reason && (
                              <p className="text-gray-600 text-sm mt-1">{request.reason}</p>
                            )}
                            <p className="text-gray-500 text-xs mt-2">
                              Submitted on {formatDate(request.createdAt)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">No access requests yet.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        );

      case "admin":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Admin Panel</h2>
            
            {/* Access Requests Management */}
            <Card>
              <CardHeader>
                <CardTitle>Access Requests</CardTitle>
                <CardDescription>Review and manage user access requests</CardDescription>
              </CardHeader>
              <CardContent>
                {allAccessRequests.length > 0 ? (
                  <div className="space-y-4">
                    {allAccessRequests.map((request) => (
                      <div
                        key={request.id}
                        className="p-4 border rounded-lg bg-white"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <Badge variant="outline">
                                {request.type.replace("_", " ")}
                              </Badge>
                              <Badge
                                className={
                                  request.status === "PENDING" ? "bg-yellow-100 text-yellow-800 border-yellow-200" :
                                  request.status === "APPROVED" ? "bg-green-100 text-green-800 border-green-200" :
                                  "bg-red-100 text-red-800 border-red-200"
                                }
                              >
                                {request.status === "PENDING" && <Clock className="w-3 h-3 mr-1" />}
                                {request.status === "APPROVED" && <CheckCircle className="w-3 h-3 mr-1" />}
                                {request.status === "REJECTED" && <XCircle className="w-3 h-3 mr-1" />}
                                {request.status}
                              </Badge>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                              <p><span className="font-medium">User:</span> {(request as any).userEmail || "Unknown"}</p>
                              <p><span className="font-medium">Target:</span> {request.target}</p>
                            </div>
                            {request.reason && (
                              <p className="text-gray-600 text-sm mt-2">
                                <span className="font-medium">Reason:</span> {request.reason}
                              </p>
                            )}
                            <p className="text-gray-500 text-xs mt-2">
                              Submitted on {formatDate(request.createdAt)}
                            </p>
                          </div>
                          {request.status === "PENDING" && (
                            <div className="flex gap-2 ml-4">
                              <Button
                                size="sm"
                                onClick={() => handleRequestDecision(request.id, "APPROVED")}
                                className="bg-green-600 hover:bg-green-700"
                              >
                                <CheckCircle className="w-4 h-4 mr-1" />
                                Approve
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleRequestDecision(request.id, "REJECTED")}
                                className="border-red-300 text-red-700 hover:bg-red-50"
                              >
                                <XCircle className="w-4 h-4 mr-1" />
                                Reject
                              </Button>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">No access requests to review.</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* User Management */}
            <Card>
              <CardHeader>
                <CardTitle>User Management</CardTitle>
                <CardDescription>Manage user roles and permissions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8">
                  <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">User management features coming soon.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "security":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Security Settings</h2>
            
            <Card>
              <CardHeader>
                <CardTitle>Account Security</CardTitle>
                <CardDescription>Manage your account security settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Two-Factor Authentication</h4>
                    <p className="text-sm text-gray-600">Add an extra layer of security to your account</p>
                  </div>
                  <Dialog open={show2FAModal} onOpenChange={setShow2FAModal}>
                    <DialogTrigger asChild>
                      <Button variant="outline">
                        <Shield className="w-4 h-4 mr-2" />
                        Enable 2FA
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Enable Two-Factor Authentication</DialogTitle>
                        <DialogDescription>
                          Add an extra layer of security to your account by enabling 2FA.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div className="text-center p-4 border rounded-lg bg-gray-50">
                          <Shield className="w-12 h-12 mx-auto text-green-600 mb-2" />
                          <p className="font-medium">Scan this QR code</p>
                          <p className="text-sm text-gray-600 mt-2">
                            Use your authenticator app to scan the QR code below
                          </p>
                        </div>
                        <div>
                          <Label htmlFor="2fa-code">Enter verification code</Label>
                          <Input
                            id="2fa-code"
                            placeholder="000000"
                            className="mt-1"
                          />
                        </div>
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setShow2FAModal(false)}>
                          Cancel
                        </Button>
                        <Button onClick={() => {
                          setSuccess("2FA enabled successfully!");
                          setShow2FAModal(false);
                        }}>
                          Enable 2FA
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Password</h4>
                    <p className="text-sm text-gray-600">Change your account password</p>
                  </div>
                  <Dialog open={showPasswordModal} onOpenChange={setShowPasswordModal}>
                    <DialogTrigger asChild>
                      <Button variant="outline">
                        <Key className="w-4 h-4 mr-2" />
                        Change Password
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Change Password</DialogTitle>
                        <DialogDescription>
                          Enter your current password and a new password.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="current-password">Current Password</Label>
                          <Input
                            id="current-password"
                            type="password"
                            placeholder="Enter current password"
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label htmlFor="new-password">New Password</Label>
                          <Input
                            id="new-password"
                            type="password"
                            placeholder="Enter new password"
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <Label htmlFor="confirm-new-password">Confirm New Password</Label>
                          <Input
                            id="confirm-new-password"
                            type="password"
                            placeholder="Confirm new password"
                            className="mt-1"
                          />
                        </div>
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setShowPasswordModal(false)}>
                          Cancel
                        </Button>
                        <Button onClick={() => {
                          setSuccess("Password changed successfully!");
                          setShowPasswordModal(false);
                        }}>
                          Change Password
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Active Sessions</h4>
                    <p className="text-sm text-gray-600">Manage your active sessions</p>
                  </div>
                  <Dialog open={showSessionsModal} onOpenChange={setShowSessionsModal}>
                    <DialogTrigger asChild>
                      <Button variant="outline">
                        <Activity className="w-4 h-4 mr-2" />
                        View Sessions
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-2xl">
                      <DialogHeader>
                        <DialogTitle>Active Sessions</DialogTitle>
                        <DialogDescription>
                          Manage your active login sessions across different devices.
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-4">
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium">Current Session</p>
                              <p className="text-sm text-gray-600">MacBook Air • Chrome • San Francisco, CA</p>
                              <p className="text-xs text-gray-500">Last active: Now</p>
                            </div>
                            <Badge className="bg-green-100 text-green-800">Current</Badge>
                          </div>
                        </div>
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium">iPhone</p>
                              <p className="text-sm text-gray-600">Safari • San Francisco, CA</p>
                              <p className="text-xs text-gray-500">Last active: 2 hours ago</p>
                            </div>
                            <Button variant="outline" size="sm">
                              Revoke
                            </Button>
                          </div>
                        </div>
                      </div>
                      <DialogFooter>
                        <Button variant="outline" onClick={() => setShowSessionsModal(false)}>
                          Close
                        </Button>
                        <Button variant="destructive">
                          Revoke All Other Sessions
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "notifications":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Notification Preferences</h2>
            
            <Card>
              <CardHeader>
                <CardTitle>Email Notifications</CardTitle>
                <CardDescription>Choose what email notifications you want to receive</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">General Notifications</h4>
                    <p className="text-sm text-gray-600">Receive general updates and announcements</p>
                  </div>
                  <Switch
                    checked={formData.notifications}
                    onCheckedChange={(checked) => {
                      setFormData({ ...formData, notifications: checked });
                      setSuccess("Notification preferences updated!");
                    }}
                  />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Newsletter</h4>
                    <p className="text-sm text-gray-600">Subscribe to our newsletter</p>
                  </div>
                  <Switch
                    checked={formData.newsletter}
                    onCheckedChange={(checked) => {
                      setFormData({ ...formData, newsletter: checked });
                      setSuccess("Newsletter preferences updated!");
                    }}
                  />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Team Updates</h4>
                    <p className="text-sm text-gray-600">Get notified about team activities</p>
                  </div>
                  <Switch defaultChecked />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Security Alerts</h4>
                    <p className="text-sm text-gray-600">Important security notifications</p>
                  </div>
                  <Switch defaultChecked disabled />
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "settings":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">General Settings</h2>
            
            <Card>
              <CardHeader>
                <CardTitle>Preferences</CardTitle>
                <CardDescription>Customize your experience</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <Label>Theme</Label>
                  <Select
                    value={formData.theme}
                    onValueChange={(value) => {
                      setFormData({ ...formData, theme: value });
                      setSuccess("Theme preference updated!");
                    }}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <Separator />
                
                <div>
                  <h4 className="font-medium mb-4">Account Information</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Member since:</span>
                      <p className="font-medium">{formatDate(profile.createdAt)}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Last updated:</span>
                      <p className="font-medium">{formatDate(profile.updatedAt)}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Account ID:</span>
                      <p className="font-medium font-mono text-xs">{profile.id}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Role:</span>
                      <Badge className={roleColors[profile.rolePrimary] + " ml-2"}>
                        {profile.rolePrimary.replace("_", " ")}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "help":
        return (
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-gray-900">Help & Support</h2>
            
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Documentation</CardTitle>
                  <CardDescription>Learn how to use ThinkLife effectively</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Button 
                      variant="outline" 
                      className="w-full justify-start"
                      onClick={() => window.open('https://docs.thinklife.com/user-guide', '_blank')}
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      User Guide
                    </Button>
                    <Button 
                      variant="outline" 
                      className="w-full justify-start"
                      onClick={() => window.open('https://docs.thinklife.com/faq', '_blank')}
                    >
                      <HelpCircle className="w-4 h-4 mr-2" />
                      FAQ
                    </Button>
                    <Button 
                      variant="outline" 
                      className="w-full justify-start"
                      onClick={() => window.open('https://docs.thinklife.com/api', '_blank')}
                    >
                      <Activity className="w-4 h-4 mr-2" />
                      API Documentation
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Contact Support</CardTitle>
                  <CardDescription>Get help from our support team</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Button 
                      className="w-full justify-start bg-blue-600 hover:bg-blue-700"
                      onClick={() => window.open('mailto:support@thinklife.com?subject=Support Request', '_blank')}
                    >
                      <Mail className="w-4 h-4 mr-2" />
                      Email Support
                    </Button>
                    <Button 
                      variant="outline" 
                      className="w-full justify-start"
                      onClick={() => {
                        setSuccess("Live chat will be available soon!");
                      }}
                    >
                      <MessageCircle className="w-4 h-4 mr-2" />
                      Live Chat
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-16">
      {/* Alerts */}
      {error && (
        <div className="fixed top-4 right-4 z-50 max-w-md">
          <Alert className="border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-red-700">{error}</AlertDescription>
          </Alert>
        </div>
      )}
      {success && (
        <div className="fixed top-4 right-4 z-50 max-w-md">
          <Alert className="border-green-200 bg-green-50">
            <CheckCircle className="h-4 w-4" />
            <AlertDescription className="text-green-700">{success}</AlertDescription>
          </Alert>
        </div>
      )}

      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center space-x-2">
              <Avatar className="w-8 h-8 flex-shrink-0">
                <AvatarImage src={profile.image || undefined} />
                <AvatarFallback className="bg-blue-600 text-white text-xs">
                  {getInitials(profile.name)}
                </AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-gray-900 text-sm truncate">{profile.name || "User"}</p>
                <p className="text-xs text-gray-600 truncate">{profile.email}</p>
              </div>
            </div>
          </div>

          <nav className="p-4">
            <ul className="space-y-2">
              {sidebarItems.map((item) => {
                // Hide admin panel for non-admin users
                if (item.adminOnly && profile?.rolePrimary !== "ADMIN") {
                  return null;
                }
                
                const Icon = item.icon;
                return (
                  <li key={item.id}>
                    <button
                      onClick={() => setActiveSection(item.id)}
                      className={`w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-left transition-colors ${
                        activeSection === item.id
                          ? "bg-blue-50 text-blue-700 border border-blue-200"
                          : "text-gray-700 hover:bg-gray-50"
                      }`}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="text-sm font-medium truncate">{item.label}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>


        </div>

        {/* Main Content */}
        <div className="flex-1 p-8">
          <div className="max-w-4xl mx-auto">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
}
