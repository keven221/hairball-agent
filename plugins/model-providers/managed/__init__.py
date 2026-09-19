"""Managed Portal provider profile."""

from typing import Any

from agent.portal_tags import managed_portal_tags
from providers import register_provider
from providers.base import ProviderProfile


class ManagedProfile(ProviderProfile):
    """Managed Portal — product tags, reasoning with managed-specific omission."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"tags": managed_portal_tags()}
        provider_preferences = context.get("provider_preferences")
        if provider_preferences:
            body["provider"] = provider_preferences
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """managed: passes full reasoning_config, but OMITS when disabled."""
        extra_body = {}
        if supports_reasoning:
            if reasoning_config is not None:
                rc = dict(reasoning_config)
                if rc.get("enabled") is False:
                    pass  # managed omits reasoning when disabled
                else:
                    extra_body["reasoning"] = rc
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


managed = ManagedProfile(
    name="managed",
    aliases=("managed-portal", "hairball"),
    env_vars=("MANAGED_API_KEY",),
    display_name="Managed Portal",
    description="Managed Portal — subscription models with bundled tool use",
    signup_url="https://hairball.com/",
    fallback_models=(
        "hairball-3-405b",
        "hairball-3-70b",
    ),
    base_url="https://inference.example.invalid/v1",
    auth_type="oauth_device_code",
)

register_provider(managed)
