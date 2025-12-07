import Keycloak from 'keycloak-js';

// Keycloak configuration
const keycloakConfig = {
  url: process.env.NEXT_PUBLIC_KEYCLOAK_URL || 'http://localhost:8080',
  realm: process.env.NEXT_PUBLIC_KEYCLOAK_REALM || 'thinklife',
  clientId: process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID || 'thinklife-frontend',
};

// Initialize Keycloak instance
export const keycloak = typeof window !== 'undefined' 
  ? new Keycloak(keycloakConfig)
  : null;

// Keycloak initialization options
export const keycloakInitOptions = {
  onLoad: 'check-sso' as const, // Check SSO on load, don't redirect if not authenticated
  checkLoginIframe: false, // Disable iframe check for better performance
  pkceMethod: 'S256' as const, // Use PKCE for better security
  enableLogging: process.env.NODE_ENV === 'development',
};

// Track initialization state
let initPromise: Promise<boolean> | null = null;
let isInitializing = false;

// Initialize Keycloak
export const initKeycloak = async (): Promise<boolean> => {
  if (!keycloak) {
    console.warn('Keycloak is not available on server side');
    return false;
  }

  // If already initialized, return the result
  if (keycloak.authenticated !== undefined) {
    return keycloak.authenticated;
  }

  // If currently initializing, wait for that promise
  if (initPromise) {
    return initPromise;
  }

  // Check if Keycloak URL is configured
  const keycloakUrl = process.env.NEXT_PUBLIC_KEYCLOAK_URL;
  if (!keycloakUrl) {
    // If not configured at all, don't try to initialize
    console.log('Keycloak not configured. Please set NEXT_PUBLIC_KEYCLOAK_URL in your .env.local file.');
    return false;
  }
  
  // Note: localhost:8080 is a valid Keycloak URL for local development
  // We should still try to initialize it

  // Start initialization
  isInitializing = true;
  initPromise = (async () => {
    try {
      const authenticated = await keycloak.init(keycloakInitOptions);
      isInitializing = false;
      return authenticated;
    } catch (error) {
      // Don't block page load if Keycloak fails - just log the error
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.warn('Keycloak initialization failed (non-blocking):', errorMessage);
      console.warn('Keycloak URL:', keycloakUrl);
      console.warn('This might be normal if Keycloak server is not running. Login will still work when user clicks login.');
      isInitializing = false;
      initPromise = null; // Reset so we can try again
      return false;
    }
  })();

  return initPromise;
};

// Ensure Keycloak is initialized (for login)
export const ensureKeycloakInitialized = async (): Promise<boolean> => {
  if (!keycloak) {
    return false;
  }

  // If already initialized, return immediately
  if (keycloak.authenticated !== undefined) {
    return true;
  }

  // Wait for initialization to complete
  await initKeycloak();
  return keycloak.authenticated !== undefined;
};

// Login function
export const login = async (redirectUri?: string) => {
  if (!keycloak) {
    console.warn('Keycloak is not available. Please configure NEXT_PUBLIC_KEYCLOAK_URL in your .env.local file.');
    return;
  }
  
  // Store the final destination in localStorage for the callback page
  const finalDestination = redirectUri || window.location.origin;
  if (typeof window !== 'undefined') {
    localStorage.setItem('keycloak_redirect_uri', finalDestination);
  }
  
  // Always use the callback page as the Keycloak redirect URI
  const keycloakRedirectUri = `${window.location.origin}/auth/callback`;
  
  const options: Keycloak.KeycloakLoginOptions = {
    redirectUri: keycloakRedirectUri,
  };
  
  try {
    // Ensure Keycloak is fully initialized before attempting login
    const isInitialized = await ensureKeycloakInitialized();
    
    if (!isInitialized) {
      // If initialization failed, check if Keycloak URL is configured
      const keycloakUrl = process.env.NEXT_PUBLIC_KEYCLOAK_URL;
      if (!keycloakUrl) {
        console.error('Keycloak is not configured. Please set NEXT_PUBLIC_KEYCLOAK_URL in your .env.local file.');
        return;
      }
      
      // If URL is configured but init failed, it might be a connection issue
      console.error('Failed to initialize Keycloak. Please check if Keycloak server is running at:', keycloakUrl);
      // Still try to login - Keycloak might handle it
      try {
        keycloak.login(options);
      } catch (loginError) {
        console.error('Login failed after initialization error:', loginError);
      }
      return;
    }
    
    // Check if already authenticated
    if (keycloak.authenticated) {
      // Already authenticated, redirect to final destination
      if (typeof window !== 'undefined') {
        window.location.href = finalDestination;
      }
      return;
    }
    
    // Now that Keycloak is initialized, trigger login
    keycloak.login(options);
  } catch (error) {
    console.error('Login failed:', error);
    // Last resort: try to login anyway
    try {
      keycloak.login(options);
    } catch (finalError) {
      console.error('Login completely failed:', finalError);
    }
  }
};

// Logout function
export const logout = (redirectUri?: string) => {
  if (!keycloak) return;
  
  const options: Keycloak.KeycloakLogoutOptions = {};
  if (redirectUri) {
    options.redirectUri = redirectUri;
  }
  
  keycloak.logout(options);
};

// Register function
export const register = (redirectUri?: string) => {
  if (!keycloak) return;
  
  const options: Keycloak.KeycloakLoginOptions = {
    action: 'REGISTER',
  };
  if (redirectUri) {
    options.redirectUri = redirectUri;
  }
  
  keycloak.login(options);
};

// Get user token
export const getToken = async (): Promise<string | undefined> => {
  if (!keycloak) return undefined;
  
  try {
    // Refresh token if needed
    await keycloak.updateToken(30); // Refresh if token expires in less than 30 seconds
    return keycloak.token;
  } catch (error) {
    console.error('Failed to get token:', error);
    return undefined;
  }
};

// Get user info
export const getUserInfo = () => {
  if (!keycloak || !keycloak.authenticated) return null;
  
  return {
    id: keycloak.tokenParsed?.sub,
    email: keycloak.tokenParsed?.email,
    name: keycloak.tokenParsed?.name,
    firstName: keycloak.tokenParsed?.given_name,
    lastName: keycloak.tokenParsed?.family_name,
    username: keycloak.tokenParsed?.preferred_username,
    roles: keycloak.realmAccess?.roles || [],
    emailVerified: keycloak.tokenParsed?.email_verified,
  };
};

// Check if user is authenticated
export const isAuthenticated = (): boolean => {
  return keycloak?.authenticated ?? false;
};

// Check if user has role
export const hasRole = (role: string): boolean => {
  if (!keycloak || !keycloak.authenticated) return false;
  return keycloak.hasRealmRole(role);
};

// Check if user has any of the specified roles
export const hasAnyRole = (roles: string[]): boolean => {
  if (!keycloak || !keycloak.authenticated) return false;
  return roles.some(role => keycloak.hasRealmRole(role));
};

