"""Credential management service for SP-API."""

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from services.constants import (
    REQUIRED_ENV,
    FIELD_LABEL,
    ENDPOINTS,
    MARKETPLACES,
    CUSTOM_MP,
    SECRET_PATHS,
)


class CredentialsManager:
    """Manages SP-API credentials from multiple sources."""
    
    def __init__(self):
        self._env_cache: Optional[Dict[str, str]] = None
        self._secrets_cache: Optional[Dict[str, str]] = None
    
    def get_env_config(self) -> Dict[str, str]:
        """Get credentials from environment variables."""
        if self._env_cache is None:
            self._env_cache = {
                k: (os.environ.get(k) or "").strip() 
                for k in REQUIRED_ENV
            }
        return self._env_cache
    
    def get_secrets_config(self) -> Dict[str, str]:
        """Get credentials from Streamlit secrets."""
        if self._secrets_cache is None:
            self._secrets_cache = {}
            for k in REQUIRED_ENV:
                val = self._get_secret_value(*SECRET_PATHS[k])
                self._secrets_cache[k] = (val or "").strip() if val else ""
        return self._secrets_cache
    
    def _get_secret_value(self, *paths: Tuple[Optional[str], str]) -> Optional[str]:
        """Try multiple secret paths."""
        for section, key in paths:
            try:
                if section:
                    val = st.secrets.get(section, {}).get(key)
                else:
                    val = st.secrets.get(key)
                if val:
                    return str(val).strip()
            except Exception:
                continue
        return None
    
    def get_base_config(self) -> Dict[str, str]:
        """Get merged config from env and secrets (env takes priority)."""
        env_cfg = self.get_env_config()
        sec_cfg = self.get_secrets_config()
        return {k: env_cfg[k] or sec_cfg[k] for k in REQUIRED_ENV}
    
    def get_session_config(self, base_cfg: Dict[str, str]) -> Dict[str, str]:
        """Get config with session overrides applied."""
        return {
            k: st.session_state.get(f"cred_{k}", base_cfg[k]) 
            for k in REQUIRED_ENV
        }
    
    def get_fingerprint(self, cfg: Dict[str, str]) -> str:
        """Generate fingerprint for cache key (excludes secrets)."""
        return hashlib.sha256(
            "|".join(cfg[k] for k in REQUIRED_ENV).encode("utf-8")
        ).hexdigest()[:12]
    
    def get_sources(
        self, 
        cfg: Dict[str, str], 
        base_cfg: Dict[str, str],
        env_cfg: Dict[str, str],
        sec_cfg: Dict[str, str]
    ) -> Dict[str, str]:
        """Determine source of each credential value."""
        sources = {}
        for k in REQUIRED_ENV:
            if not cfg[k]:
                sources[k] = "—"
            elif cfg[k] != base_cfg[k]:
                sources[k] = "панель"
            elif env_cfg[k]:
                sources[k] = ".env"
            else:
                sources[k] = "secrets" if sec_cfg[k] else "панель"
        return sources
    
    def validate_complete(self, cfg: Dict[str, str]) -> List[str]:
        """Return list of missing required fields."""
        return [FIELD_LABEL[k] for k in REQUIRED_ENV if not cfg[k]]
    
    def mask_value(self, value: str, keep: int = 4) -> str:
        """Mask a secret value for display."""
        if not value:
            return "—"
        if len(value) <= keep:
            return "•" * len(value)
        return f"{'•' * 8}{value[-keep:]}"
    
    def render_sidebar(self) -> Tuple[Dict[str, str], str]:
        """Render credentials sidebar and return (config, fingerprint)."""
        env_cfg = self.get_env_config()
        sec_cfg = self.get_secrets_config()
        base_cfg = self.get_base_config()
        
        with st.sidebar:
            st.subheader("🔐 Доступи SP-API")
            
            cfg = self.get_session_config(base_cfg)
            filled = [k for k in REQUIRED_ENV if cfg[k]]
            
            if len(filled) == len(REQUIRED_ENV):
                badge, text = "cred-ok", "готово до запиту"
            elif filled:
                badge, text = "cred-partial", f"заповнено {len(filled)} з {len(REQUIRED_ENV)}"
            else:
                badge, text = "cred-missing", "не заповнено"
            
            st.markdown(
                f'<span class="cred-badge {badge}">{text}</span>', 
                unsafe_allow_html=True
            )
            
            self._render_source_info(env_cfg, sec_cfg)
            
            with st.expander("Змінити доступи", expanded=not filled):
                self._render_credential_inputs(cfg)
                
                # Region selection
                region_names = list(ENDPOINTS)
                current_ep = cfg["SP_API_ENDPOINT"]
                ep_index = next(
                    (i for i, n in enumerate(region_names) if ENDPOINTS[n] == current_ep), 
                    0
                )
                region = st.selectbox("Регіон", region_names, index=ep_index, key="cred_region")
                
                # Marketplace selection
                mp_options = list(MARKETPLACES) + [CUSTOM_MP]
                current_mp = cfg["SP_API_MARKETPLACE_ID"]
                mp_index = mp_options.index(current_mp) if current_mp in MARKETPLACES else (
                    len(mp_options) - 1 if current_mp else 0
                )
                mp_choice = st.selectbox(
                    "Маркетплейс",
                    mp_options,
                    index=mp_index,
                    format_func=lambda m: MARKETPLACES.get(m, m),
                    key="cred_marketplace_choice",
                )
                
                if mp_choice == CUSTOM_MP:
                    marketplace = st.text_input(
                        FIELD_LABEL["SP_API_MARKETPLACE_ID"],
                        value=current_mp if current_mp not in MARKETPLACES else "",
                        key="cred_marketplace_custom",
                        placeholder="ATVPDKIKX0DER",
                    ).strip()
                else:
                    marketplace = mp_choice
                
                self._render_credentials_help()
                
                # Update config with selections
                cfg["SP_API_ENDPOINT"] = ENDPOINTS[region]
                cfg["SP_API_MARKETPLACE_ID"] = marketplace
                cfg = {k: (v or "").strip() for k, v in cfg.items()}
            
            # Display current values
            sources = self.get_sources(cfg, base_cfg, env_cfg, sec_cfg)
            self._render_current_values(cfg, sources)
            
            # Test connection button
            if st.button("Перевірити з'єднання", use_container_width=True):
                self._test_connection(cfg)
        
        fingerprint = self.get_fingerprint(cfg)
        return cfg, fingerprint
    
    def _render_source_info(self, env_cfg: Dict[str, str], sec_cfg: Dict[str, str]) -> None:
        """Render info about credential sources."""
        picked = []
        if any(env_cfg.values()):
            picked.append(f"<code>.env</code> — {sum(1 for v in env_cfg.values() if v)}")
        if any(sec_cfg.values()):
            picked.append(f"<code>secrets</code> — {sum(1 for v in sec_cfg.values() if v)}")
        if picked:
            st.markdown(
                '<div class="cred-source">Підхоплено з ' + ", ".join(picked)
                + f" (усього полів {len(REQUIRED_ENV)})</div>",
                unsafe_allow_html=True,
            )
    
    def _render_credential_inputs(self, cfg: Dict[str, str]) -> None:
        """Render credential input fields."""
        st.text_input(
            FIELD_LABEL["SP_API_LWA_CLIENT_ID"],
            value=cfg["SP_API_LWA_CLIENT_ID"],
            key="cred_SP_API_LWA_CLIENT_ID",
            placeholder="amzn1.application-oa2-client....",
        )
        st.text_input(
            FIELD_LABEL["SP_API_LWA_CLIENT_SECRET"],
            value=cfg["SP_API_LWA_CLIENT_SECRET"],
            key="cred_SP_API_LWA_CLIENT_SECRET",
            type="password",
        )
        st.text_input(
            FIELD_LABEL["SP_API_REFRESH_TOKEN"],
            value=cfg["SP_API_REFRESH_TOKEN"],
            key="cred_SP_API_REFRESH_TOKEN",
            type="password",
            placeholder="Atzr|....",
        )
    
    def _render_credentials_help(self) -> None:
        """Render help text for credentials."""
        st.caption(
            "Введене тут живе тільки в цій сесії й на диск не пишеться. "
            "Щоб зберегти назавжди — `.env` поруч з `app.py` (локально) "
            "або `[sp_api]` у Streamlit secrets (на деплої):\n\n"
            "```toml\n[sp_api]\nlwa_client_id = \"...\"\n"
            "lwa_client_secret = \"...\"\nrefresh_token = \"...\"\n"
            "marketplace_id = \"ATVPDKIKX0DER\"\n"
            "endpoint = \"https://sellingpartnerapi-na.amazon.com\"\n```"
        )
    
    def _render_current_values(
        self, 
        cfg: Dict[str, str], 
        sources: Dict[str, str]
    ) -> None:
        """Render current credential values table."""
        import pandas as pd
        
        st.markdown("**Поточні значення**")
        st.dataframe(
            pd.DataFrame([
                {
                    "Поле": FIELD_LABEL[k],
                    "Значення": cfg[k] if k in
                    ("SP_API_MARKETPLACE_ID", "SP_API_ENDPOINT") else self.mask_value(cfg[k]),
                    "Джерело": sources[k],
                }
                for k in REQUIRED_ENV
            ]),
            hide_index=True,
            use_container_width=True,
        )
    
    def _test_connection(self, cfg: Dict[str, str]) -> None:
        """Test SP-API connection."""
        from services.data_loader import get_access_token
        from services.exceptions import AuthenticationError
        
        missing_now = [FIELD_LABEL[k] for k in REQUIRED_ENV if not cfg[k]]
        if missing_now:
            st.error("Не заповнено: " + ", ".join(missing_now))
            return
        
        try:
            token = get_access_token(cfg)
            st.success(f"LWA токен отримано ({self.mask_value(token, 6)})")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# Global instance
credentials_manager = CredentialsManager()


def get_credentials_panel() -> Tuple[Dict[str, str], str]:
    """Convenience function to get credentials panel."""
    return credentials_manager.render_sidebar()