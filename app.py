from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Hobbii AutoStore Data Explorer', page_icon='📦', layout='wide')
st.title('📦 Hobbii AutoStore Data Explorer')
st.caption('V2.4 Cloud Parquet Edition | AutoStore × Planday | Theoretical vs Working-hour OLPH')

DATA_DIR = Path(__file__).resolve().parent / 'data' / 'processed'
AUTO_FILE = DATA_DIR / 'autostore.parquet'
PLANDAY_FILE = DATA_DIR / 'planday.parquet'
for data_file in (AUTO_FILE, PLANDAY_FILE):
    if not data_file.is_file():
        st.error(f'Missing data file: {data_file}')
        st.stop()

@st.cache_data(show_spinner='Loading AutoStore and Planday Parquet files...')
def load_data(auto_path, auto_mtime, pd_path, pd_mtime):
    auto_columns = ['date', 'start_time', 'port_id', 'task_type', 'bin_count',
                    'sku_count', 'time_spent_total', 'time_spent_user']
    a = pd.read_parquet(auto_path, columns=auto_columns)
    p = pd.read_parquet(pd_path)
    return a, p

auto, pd_raw = load_data(str(AUTO_FILE), AUTO_FILE.stat().st_mtime_ns,
                         str(PLANDAY_FILE), PLANDAY_FILE.stat().st_mtime_ns)
for c in ['date', 'start_time']:
    auto[c] = pd.to_datetime(auto[c], errors='coerce')
for c in ['port_id', 'bin_count', 'sku_count', 'time_spent_total', 'time_spent_user']:
    auto[c] = pd.to_numeric(auto[c], errors='coerce')
auto['task_type'] = auto['task_type'].astype('string').fillna('(Missing)')
# Business-defined Pick Type from bin_count (do not rely on source task_type labels).
auto['pick_type_by_bins'] = pd.Series(pd.NA, index=auto.index, dtype='string')
auto.loc[auto['bin_count'].between(1, 2), 'pick_type_by_bins'] = 'Small Orders'
auto.loc[auto['bin_count'].between(3, 6), 'pick_type_by_bins'] = 'Normal Orders'
auto.loc[auto['bin_count'].ge(7), 'pick_type_by_bins'] = 'Large Orders'
auto = auto.loc[auto['port_id'].between(101, 112) & auto['date'].notna() & auto['start_time'].notna()].copy()
auto['port_id'] = auto['port_id'].astype(int)
for c in ['bin_count', 'sku_count', 'time_spent_total', 'time_spent_user']:
    auto[c] = auto[c].where(auto[c] >= 0)

pd_raw.columns = pd_raw.columns.str.strip().str.lower().str.replace(' ', '_')
required_pd = {'date', 'station', 'start_time', 'end_time', 'hours', 'shift_type'}
if not required_pd.issubset(pd_raw.columns):
    st.error(f'Missing Planday columns: {sorted(required_pd - set(pd_raw.columns))}')
    st.stop()
# Only Planday shifts explicitly labelled Picking are eligible.
pd_raw = pd_raw.loc[
    pd_raw['shift_type'].astype('string').str.strip().str.casefold().eq('picking').fillna(False)
].copy()
pd_raw['date'] = pd.to_datetime(pd_raw['date'], errors='coerce').dt.normalize()
# Use the station code at the end of Position, when available.
if 'position' in pd_raw.columns:
    station_from_position = pd.to_numeric(
        pd_raw['position'].astype('string').str.extract(r'(10[1-9]|11[0-2])\s*$')[0],
        errors='coerce'
    )
    pd_raw['station'] = station_from_position.fillna(pd.to_numeric(pd_raw['station'], errors='coerce'))
else:
    pd_raw['station'] = pd.to_numeric(pd_raw['station'], errors='coerce')
pd_raw['hours'] = pd.to_numeric(pd_raw['hours'], errors='coerce')

# Excel time cells may be stored as text, Python time, or timedelta-like strings.
def clock_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value in ('', 'NaT', 'None', 'nan'):
        return None
    if ' ' in value:
        value = value.split(' ')[-1]
    return value

pd_raw['start_clock'] = pd_raw['start_time'].map(clock_text)
pd_raw['end_clock'] = pd_raw['end_time'].map(clock_text)
pd_raw['shift_start'] = pd.to_datetime(pd_raw['date'].dt.strftime('%Y-%m-%d') + ' ' + pd_raw['start_clock'], errors='coerce')
pd_raw['shift_end'] = pd.to_datetime(pd_raw['date'].dt.strftime('%Y-%m-%d') + ' ' + pd_raw['end_clock'], errors='coerce')
pd_raw.loc[pd_raw['shift_end'] <= pd_raw['shift_start'], 'shift_end'] += pd.Timedelta(days=1)
pd_raw['working_hours'] = pd_raw['hours']
# Confirmed Hobbii evening-shift picking window: 15:45–23:40, excluding 50 min breaks and final 5 min.
evening = (pd_raw['start_clock'].str.startswith('15:45', na=False) & pd_raw['end_clock'].str.startswith('23:45', na=False))
pd_raw.loc[evening, 'working_hours'] = 425 / 60
# The final five minutes are not part of effective picking time.
pd_raw.loc[evening, 'shift_end'] = (
    pd_raw.loc[evening, 'date'] + pd.Timedelta(hours=23, minutes=40)
)
pd_raw = pd_raw.loc[
    pd_raw['station'].between(101, 112) & pd_raw['date'].notna() &
    pd_raw['shift_start'].notna() & pd_raw['shift_end'].notna() &
    pd_raw['working_hours'].gt(0)
].copy()
pd_raw['station'] = pd_raw['station'].astype(int)

# Same signature exported in different files: retain maximum multiplicity found in any ONE file.
# This is an approximation: without employee identifiers, duplicate exports cannot
# always be distinguished from multiple employees on the same station/shift.
# Without employee identifiers, this is a cautious heuristic, not verified deduplication.
keys = ['date', 'station', 'shift_start', 'shift_end', 'hours', 'position'] if 'position' in pd_raw.columns else ['date', 'station', 'shift_start', 'shift_end', 'hours']
if 'source_file' in pd_raw.columns:
    pd_raw['_occurrence'] = pd_raw.groupby(keys + ['source_file'], dropna=False).cumcount()
    pd_raw = pd_raw.drop_duplicates(subset=keys + ['_occurrence']).drop(columns='_occurrence')

st.sidebar.header('Filters')
if auto.empty:
    st.error('No valid AutoStore records found. Check station, date, and time columns.')
    st.stop()
min_date, max_date = auto['date'].min().date(), auto['date'].max().date()
date_range = st.sidebar.date_input('Select Date Range', (min_date, max_date), min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.info('Select both start and end dates.')
    st.stop()
start_date, end_date = date_range
stations = st.sidebar.multiselect('Select Stations', list(range(101, 113)), default=list(range(101, 113)))
all_types = ['Small Orders', 'Normal Orders', 'Large Orders']
types = st.sidebar.multiselect('Select Pick Types', all_types, default=all_types)
unit = st.sidebar.selectbox('AutoStore time unit (verify)', ['Seconds', 'Milliseconds'])
factor = 1.0 if unit == 'Seconds' else 0.001

start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)
base = auto.loc[auto['date'].ge(start_ts) & auto['date'].lt(end_ts) & auto['port_id'].isin(stations) & auto['pick_type_by_bins'].notna()].copy()
selected = base.loc[base['pick_type_by_bins'].isin(types)].copy()
shifts = pd_raw.loc[pd_raw['date'].ge(start_ts) & pd_raw['date'].lt(end_ts) & pd_raw['station'].isin(stations)].copy()
if selected.empty:
    st.warning('No AutoStore records for these filters.')
    st.stop()

# Match a task by station AND actual shift interval. A task can count only once.
# For standard evening shifts, exclude both breaks and the final five minutes.
# A shift contributes its working hours only if it contains at least one AutoStore task.
@st.cache_data(show_spinner='Matching tasks to Planday shift windows...')
def match_shifts(tasks, shifts):
    task_matched = np.zeros(len(tasks), dtype=bool)
    shift_matched = np.zeros(len(shifts), dtype=bool)
    if tasks.empty or shifts.empty:
        return task_matched, shift_matched
    t = tasks.reset_index(drop=True)
    s = shifts.reset_index(drop=True)
    task_groups = t.groupby(['port_id', t['date'].dt.normalize()]).indices
    shift_groups = s.groupby(['station', 'date']).indices
    for key, shift_positions in shift_groups.items():
        task_positions = task_groups.get(key)
        if task_positions is None:
            continue
        timestamps = t.iloc[task_positions]['start_time'].to_numpy(dtype='datetime64[ns]')
        for j in shift_positions:
            shift = s.iloc[j]
            mask = (timestamps >= np.datetime64(shift['shift_start'])) & (timestamps < np.datetime64(shift['shift_end']))
            if (str(shift['start_clock']).startswith('15:45') and
                    str(shift['end_clock']).startswith('23:45')):
                day = np.datetime64(shift['date'])
                mask &= ~((timestamps >= day + np.timedelta64(19 * 60, 'm')) &
                          (timestamps < day + np.timedelta64(19 * 60 + 30, 'm')))
                mask &= ~((timestamps >= day + np.timedelta64(21 * 60 + 40, 'm')) &
                          (timestamps < day + np.timedelta64(22 * 60, 'm')))
            if mask.any():
                task_matched[task_positions[mask]] = True
                shift_matched[j] = True
    return task_matched, shift_matched

# Work-time OLPH is computed for all picking types; a type-only filter cannot legitimately
# reuse the whole shift denominator for one category.
matched_tasks, matched_shifts = match_shifts(base, shifts)
base = base.reset_index(drop=True)
shifts = shifts.reset_index(drop=True)
base['matched'] = matched_tasks
shifts['matched'] = matched_shifts
selected = base.loc[base['pick_type_by_bins'].isin(types)].copy()
all_types_selected = set(types) == set(all_types)

# Only matched tasks belong in the numerator of Working-hour OLPH.
matched_base = base.loc[base['matched']].copy()
matched_selected = selected.loc[selected['matched']].copy()
used_shifts = shifts.loc[shifts['matched']].copy()

def summarize_tasks(frame):
    return pd.Series({
        'Records': len(frame),
        'Total_Bins': frame['bin_count'].sum(),
        'Total_SKU': frame['sku_count'].sum(),
        'Task_Hours': frame['time_spent_total'].sum() * factor / 3600,
        'User_Hours': frame['time_spent_user'].sum() * factor / 3600,
    })

def safe_ratio(n, d):
    return n / d if pd.notna(d) and d > 0 else np.nan

all_stats = summarize_tasks(selected)
matched_stats = summarize_tasks(matched_selected)
worked_hours = used_shifts['working_hours'].sum()
theoretical = safe_ratio(all_stats['Total_Bins'], all_stats['Task_Hours'])
actual = safe_ratio(matched_stats['Total_Bins'], worked_hours) if all_types_selected else np.nan
coverage = safe_ratio(len(matched_base), len(base)) * 100

st.subheader('Performance Overview')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Theoretical OLPH*', f'{theoretical:,.1f}' if pd.notna(theoretical) else 'N/A')
c2.metric('Actual OLPH*', f'{actual:,.1f}' if pd.notna(actual) else 'N/A')
c3.metric('Difference (Theoretical − Actual)', f'{theoretical - actual:,.1f}' if pd.notna(actual) and pd.notna(theoretical) else 'N/A')
c4.metric('Matched Task Coverage', f'{coverage:.1f}%' if pd.notna(coverage) else 'N/A')
c5, c6, c7, c8 = st.columns(4)
c5.metric('AutoStore Task Hours', f"{all_stats['Task_Hours']:,.1f}")
c6.metric('AutoStore User Hours', f"{all_stats['User_Hours']:,.1f}")
c7.metric('Matched Planday Working Hours', f'{worked_hours:,.1f}')
c8.metric('Picking Records (selected)', f'{len(selected):,}')
st.caption(f'Dates: {start_date} – {end_date} | Matched Picking shifts: {len(used_shifts):,} | Planday Picking shift records after heuristic deduplication: {len(pd_raw):,}')
st.info('Pick Types are defined by bin_count: Small 1–2, Normal 3–6, Large 7+. Records with missing/zero bins are excluded. OLPH* uses bin_count as an ORDERLINE PROXY, not verified orderline counts. Theoretical OLPH uses all selected task records; Actual OLPH uses only tasks within Planday shift windows and working hours from matched shifts. Their difference is descriptive, not a causal measure of lost productivity.')
if all_types_selected and used_shifts.empty:
    st.warning('No matching Planday Picking shifts found for this date/station selection; Actual OLPH is unavailable.')
if not all_types_selected:
    st.warning('Actual OLPH is hidden when only some Pick Types are selected: Planday does not identify working hours spent on each Pick Type. Select all Pick Types for a valid station/shift-level comparison.')
st.warning('Only Planday Shift type = Picking is included. Planday has no employee ID. Identical shifts across files are heuristically deduplicated; same-time shifts for multiple employees cannot be verified. Working-hour OLPH is provisional until shift multiplicity and break handling are validated.')

# Station summaries: use matched tasks for the actual OLPH numerator.
station_task = selected.groupby('port_id').agg(Records=('port_id', 'size'), Bins=('bin_count', 'sum'), SKU=('sku_count', 'sum'), Task_Seconds=('time_spent_total', 'sum')).reset_index()
station_task['Task_Hours'] = station_task['Task_Seconds'] * factor / 3600
station_task['Theoretical_OLPH'] = station_task['Bins'] / station_task['Task_Hours'].replace(0, np.nan)
station_matched = matched_selected.groupby('port_id').agg(Matched_Bins=('bin_count', 'sum'), Matched_Records=('port_id', 'size')).reset_index()
station_work = used_shifts.groupby('station').agg(Working_Hours=('working_hours', 'sum'), Matched_Shifts=('station', 'size')).reset_index().rename(columns={'station': 'port_id'})
station_summary = station_task.merge(station_matched, on='port_id', how='left').merge(station_work, on='port_id', how='left')
station_summary['Actual_OLPH'] = station_summary['Matched_Bins'] / station_summary['Working_Hours'].replace(0, np.nan) if all_types_selected else np.nan
station_summary['Difference'] = station_summary['Theoretical_OLPH'] - station_summary['Actual_OLPH']
station_summary['Station'] = station_summary['port_id'].astype(str)

with st.expander('Data Validation — Picking shifts and matching'):
    st.write({
        'Planday Picking shifts in selected date/station range': int(len(shifts)),
        'Matched Picking shifts': int(len(used_shifts)),
        'Matched AutoStore records (all Pick Types)': int(len(matched_base)),
        'Selected AutoStore records': int(len(selected)),
        'Matched AutoStore records (selected Pick Types)': int(len(matched_selected)),
        'All Pick Types selected': bool(all_types_selected),
    })
    st.caption('Shift deduplication is heuristic. Task matching uses task start_time and station. Overlapping shifts and multiple employees may bias working-hour totals.')

st.subheader('Station Comparison')
metric_map = {'Theoretical OLPH*': 'Theoretical_OLPH', 'Actual OLPH*': 'Actual_OLPH', 'Difference': 'Difference', 'Task Hours': 'Task_Hours', 'Working Hours': 'Working_Hours', 'Total Bins': 'Bins'}
metric = st.selectbox('Compare metric', list(metric_map))
fig = px.bar(station_summary, x='Station', y=metric_map[metric], title=f'{metric} by Station', text_auto='.1f')
fig.update_layout(xaxis_type='category')
st.plotly_chart(fig, use_container_width=True)

# Daily summary
daily_task = selected.assign(Day=selected['date'].dt.normalize()).groupby(['Day', 'port_id']).agg(Bins=('bin_count', 'sum'), Task_Seconds=('time_spent_total', 'sum')).reset_index()
daily_task['Task_Hours'] = daily_task['Task_Seconds'] * factor / 3600
daily_task['Theoretical_OLPH'] = daily_task['Bins'] / daily_task['Task_Hours'].replace(0, np.nan)
daily_matched = matched_selected.assign(Day=matched_selected['date'].dt.normalize()).groupby(['Day', 'port_id']).agg(Matched_Bins=('bin_count', 'sum')).reset_index()
daily_work = used_shifts.groupby(['date', 'station']).agg(Working_Hours=('working_hours', 'sum')).reset_index().rename(columns={'date': 'Day', 'station': 'port_id'})
daily = daily_task.merge(daily_matched, on=['Day', 'port_id'], how='left').merge(daily_work, on=['Day', 'port_id'], how='left')
daily['Actual_OLPH'] = daily['Matched_Bins'] / daily['Working_Hours'].replace(0, np.nan) if all_types_selected else np.nan
daily['Difference'] = daily['Theoretical_OLPH'] - daily['Actual_OLPH']
daily['Station'] = daily['port_id'].astype(str)
st.subheader('📈 Daily Performance Trend')
daily_metric = st.selectbox('Daily metric', list(metric_map), key='daily_metric')
fig_daily = px.line(daily, x='Day', y=metric_map[daily_metric], color='Station', markers=True, title=f'Daily {daily_metric} by Station')
fig_daily.update_layout(hovermode='x unified')
st.plotly_chart(fig_daily, use_container_width=True)

st.subheader('Station Summary Table')
show_cols = ['Station', 'Records', 'Bins', 'Task_Hours', 'Matched_Records', 'Matched_Bins', 'Working_Hours', 'Theoretical_OLPH', 'Actual_OLPH', 'Difference']
st.dataframe(station_summary[show_cols].round(2), use_container_width=True, hide_index=True)
with st.expander('Daily Summary Table'):
    st.dataframe(daily.round(2), use_container_width=True, hide_index=True)
st.download_button('📥 Download Station Summary', station_summary[show_cols].to_csv(index=False).encode('utf-8-sig'), f'autostore_planday_{start_date}_{end_date}.csv', 'text/csv')
st.caption('Pick Type derived from bin_count: Small 1–2; Normal 3–6; Large 7+. Planday Shift type = Picking only. Prototype for operational analysis. Validate Planday duplicate shifts, work-hour conventions, station staffing, and the bin_count/orderline relationship before using Actual OLPH for management decisions.')
