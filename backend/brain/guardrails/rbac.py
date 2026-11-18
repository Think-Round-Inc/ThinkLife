"""
Role-Based Access Control (RBAC) for ThinkLife agent tools.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class RBACManager:
    """
    Manages role-based authorization for agent tools.

    Security model:
    - Deny by default
    - Only explicitly allowed role/tool combinations are permitted
    """

    def __init__(self):
        self.role_tool_permissions: Dict[str, List[str]] = {
            "thinklife_admin": [
                "tavily_search",
            ],
            "thinklife_staff": [
                "tavily_search",
            ],
            "thinklife_user": [
                "tavily_search",
            ],
        }

    def is_tool_allowed(
        self,
        roles: List[str],
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Check whether any of the user's roles is allowed
        to use the requested tool.
        """

        if not roles:
            return {
                "allowed": False,
                "reason": "No roles assigned",
                "matched_role": None,
            }

        if not tool_name:
            return {
                "allowed": False,
                "reason": "Tool name missing",
                "matched_role": None,
            }

        normalized_roles = [
            role.strip().lower()
            for role in roles
            if isinstance(role, str) and role.strip()
        ]

        for role in normalized_roles:
            allowed_tools = self.role_tool_permissions.get(role, [])

            if tool_name in allowed_tools:
                return {
                    "allowed": True,
                    "reason": "Tool access permitted",
                    "matched_role": role,
                }

        return {
            "allowed": False,
            "reason": "Role is not authorized for this tool",
            "matched_role": None,
        }
    #STEP 7: RBAC Audit logging
    def log_authorization_event(
            self,
            user_id: str,
            roles: List[str],
            tool_name: str,
            allowed: bool,
            reason: str
    ) -> None:
        """
        Log RBAC authorization decisions without logging
        sensitive user content.
        """
        event = "RBAC_ALLOW" if allowed else "RBAC_DENY"
        log_method = logger.info if allowed else logger.warning
        log_method(
            "%s user=%s roles=%s tool=%s reason=%s",
            event,
            user_id,
            roles,
            tool_name,
            reason,
        )
            
   

_rbac_manager_instance = None


def get_rbac_manager() -> RBACManager:
    """Return the shared RBAC manager instance."""
    global _rbac_manager_instance

    if _rbac_manager_instance is None:
        _rbac_manager_instance = RBACManager()

    return _rbac_manager_instance