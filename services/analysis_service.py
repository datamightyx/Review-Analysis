"""Main service orchestrating Returns Analysis workflow."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, List, Optional, Tuple

import pandas as pd
import streamlit as st

from services.constants import (
    DATE_PRESETS,
    DEFAULT_DATE_RANGE_DAYS,
    DEFAULT_RELEASE_LAG_DAYS,
    MAX_RELEASE_LAG_DAYS,
    CACHE_TTL_SECONDS,
)
from services.credentials import CredentialsManager, get_credentials_panel
from services.data_loader import (
    fetch_customer_returns,
    parse_customer_returns,
    fetch_refund_events,
    refunds_to_return_rows,
    validate_date_range,
)
from services.excel_builder import build_workbook, workbook_to_bytes
from services.exceptions import (
    ReturnsAnalysisError,
    AuthenticationError,
    APIError,
    RateLimitError,
    ReportError,
    DataError,
    ValidationError,
)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def _load_returns(cfg_fp: str, start_s: str, end_s: str, _cfg: dict) -> str:
    """Cached returns report. cfg_fp is the cache key; _cfg is not hashed."""
    return fetch_customer_returns(_cfg, start_s, end_s)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def _load_refunds(
    cfg_fp: str, start_s: str, end_s: str, lag_days: int, _cfg: dict
) -> pd.DataFrame:
    """Cached refund events. cfg_fp is the cache key; _cfg is not hashed."""
    return fetch_refund_events(_cfg, start_s, end_s, release_lag_days=lag_days)


@dataclass
class AnalysisConfig:
    """Configuration for analysis run."""
    start_date: date
    end_date: date
    with_refunds: bool
    release_lag_days: int
    marketplace_id: str


@dataclass
class AnalysisResult:
    """Result of analysis run."""
    df: pd.DataFrame
    n_returns: int
    n_refunds_added: int
    skipped_skus: List[str]
    workbook_bytes: bytes


class ReturnsAnalysisService:
    """Main service for Returns Analysis workflow."""
    
    def __init__(self):
        self.credentials_manager = CredentialsManager()
    
    def render_credentials(self) -> Tuple[dict, str]:
        """Render credentials panel."""
        return get_credentials_panel()
    
    def render_date_selection(self) -> Tuple[date, date, bool, int]:
        """Render date selection UI."""
        today = date.today()
        
        st.markdown("### 📅 Період аналізу")
        
        # Date presets
        preset_cols = st.columns(len(DATE_PRESETS))
        for i, (label, days) in enumerate(DATE_PRESETS.items()):
            with preset_cols[i]:
                if st.button(label, use_container_width=True, key=f"preset_{days}"):
                    st.session_state["date_start"] = today - timedelta(days=days)
                    st.session_state["date_end"] = today
                    st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "Початок періоду",
                value=st.session_state.get("date_start", today - timedelta(days=DEFAULT_DATE_RANGE_DAYS)),
                max_value=today,
                key="date_start_input",
            )
        with col2:
            end = st.date_input(
                "Кінець періоду",
                value=st.session_state.get("date_end", today),
                max_value=today,
                key="date_end_input",
            )
        
        # Update session state
        st.session_state["date_start"] = start
        st.session_state["date_end"] = end
        
        with_refunds = st.checkbox(
            "Додати refund-и без фізичного повернення (Finances API)",
            value=True,
            help="Звіт повернень містить лише те, що фізично приїхало на FC. Refund-и "
                 "включають returnless refunds, товар, який покупець не відправив назад, "
                 "і повернення ще в дорозі.",
            key="with_refunds",
        )
        
        lag = st.slider(
            "Запас на пізній реліз транзакцій, днів",
            min_value=0,
            max_value=MAX_RELEASE_LAG_DAYS,
            value=DEFAULT_RELEASE_LAG_DAYS,
            disabled=not with_refunds,
            help="Finances API датує refund моментом РЕЛІЗУ транзакції, а не моментом "
                 "повернення грошей — реліз відстає на кілька днів. Без запасу останній "
                 "тиждень періоду недорахується. Зворотний бік: у вибірку потраплять "
                 "refund-и, зроблені вже після кінця періоду, але релізнуті в ці дні.",
            key="release_lag",
        )
        
        return start, end, with_refunds, lag
    
    def validate_inputs(self, start: date, end: date) -> None:
        """Validate user inputs."""
        if start > end:
            raise ValidationError("Початок періоду пізніше за кінець.")
        if (end - start).days > 180:
            st.warning("Finances API віддає щонайбільше ~180 днів історії. Звузь діапазон.")
    
    def run_analysis(
        self,
        cfg: dict,
        start: date,
        end: date,
        with_refunds: bool,
        release_lag_days: int,
        progress_callback: Optional[Callable] = None,
        cfg_fp: Optional[str] = None,
    ) -> AnalysisResult:
        """Run the complete analysis workflow."""
        start_s, end_s = start.isoformat(), end.isoformat()
        cfg_fp = cfg_fp or self.credentials_manager.get_fingerprint(cfg)
        
        # Fetch returns report
        if progress_callback:
            progress_callback("Створення й очікування звіту GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA", 0)
        
        try:
            raw = _load_returns(cfg_fp, start_s, end_s, cfg)
        except Exception as e:
            raise ReportError(f"Помилка звіту повернень: {e}") from e
        
        df = parse_customer_returns(raw)
        if df.empty:
            raise DataError("За цей період повернень немає.")
        
        if progress_callback:
            progress_callback(f"Отримано {len(df)} повернень, {df['asin'].nunique()} ASIN", 50)
        
        # Fetch refunds if enabled
        n_refund_added = 0
        skipped_skus = []
        
        if with_refunds:
            if progress_callback:
                progress_callback("Завантаження refund-ів (Finances API)...", 50)
            
            try:
                ref = _load_refunds(cfg_fp, start_s, end_s, release_lag_days, cfg)
            except Exception as e:
                raise APIError(f"Помилка Finances API: {e}") from e
            
            if progress_callback:
                progress_callback(f"Отримано {len(ref)} refund-позицій, {int(ref['quantity'].sum())} юнітів", 75)
            
            extra, skipped_skus = refunds_to_return_rows(ref, df, verbose=False)
            n_refund_added = len(extra)
            df = pd.concat([df, extra], ignore_index=True)
            
            if progress_callback:
                progress_callback(f"Refund-и без повернення: {n_refund_added}", 90)
        
        # Build workbook
        if progress_callback:
            progress_callback("Побудова звіту...", 95)
        
        wb = build_workbook(df)
        workbook_bytes = workbook_to_bytes(wb)
        
        n_returns = len(df) - n_refund_added
        
        return AnalysisResult(
            df=df,
            n_returns=n_returns,
            n_refunds_added=n_refund_added,
            skipped_skus=skipped_skus,
            workbook_bytes=workbook_bytes,
        )
    
    def render_results(self, result: AnalysisResult, start: date, end: date) -> None:
        """Render analysis results."""
        st.caption(f"Період: **{start} .. {end}** (UTC), {(end - start).days + 1} днів")
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всього кейсів", len(result.df))
        c2.metric("Фізичні повернення", result.n_returns)
        c3.metric("Refund без повернення", result.n_refunds_added)
        c4.metric("ASIN", result.df["asin"].nunique())
        
        # Info messages
        if result.n_refunds_added:
            st.info(
                f"Додано **{result.n_refunds_added}** refund-ів без фізичного повернення. "
                "Коментарів у них немає за визначенням, тому mismatch-аналіз рахується "
                "лише по фізичних поверненнях. Частина з них — повернення ще в дорозі "
                "(вікно ~30 днів), тому при повторному запуску пізніше цифра зменшиться."
            )
        
        if result.skipped_skus:
            st.caption(
                f"SKU без жодного повернення (не змаплені на ASIN, пропущені): {result.skipped_skus}"
            )
        
        # Download button
        file_name = f"Returns_Analysis_{start.isoformat()}_{end.isoformat()}.xlsx"
        st.download_button(
            label="⬇ Завантажити результат (.xlsx)",
            data=result.workbook_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )


def create_service() -> ReturnsAnalysisService:
    """Factory function to create service instance."""
    return ReturnsAnalysisService()