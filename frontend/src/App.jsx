import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Area, LineChart, Line, ComposedChart, ReferenceLine,
  BarChart, Bar, LabelList,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts';
import { getLocations, simulate, aggregate, stability, money, dataQuality } from './api.js';
import { FaGithub } from 'react-icons/fa';
import { GiMoon } from 'react-icons/gi';
import { FaRegFileAlt } from 'react-icons/fa';

const TRANSPOSITION_MODELS = ['perez', 'haydavies', 'isotropic'];

// The backend ships `timestamp_local` as NZ (Pacific/Auckland) wall time
// "YYYY-MM-DD HH:MM", so these formatters just slice/prettify it — no
// client-side timezone conversion needed (and none that could silently drift).
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function formatTick(v) {          // axis tick -> "21 Dec 2020, 13:00"
  if (!v) return v;
  const [date, time] = v.split(' ');
  const [y, m, d] = date.split('-');
  return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}, ${time}`;
}

function formatLocalFull(v) {     // tooltip -> "21 Dec 2020, 13:00"
  if (!v) return '';
  const [date, time] = v.split(' ');
  const [y, m, d] = date.split('-');
  return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}, ${time}`;
}

const NZ_TZ = 'Pacific/Auckland';

// UTC offset (ms) of the NZ timezone at a given instant, via Intl.
function zonedOffsetMs(ms) {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: NZ_TZ, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const p = {};
  for (const part of dtf.formatToParts(new Date(ms))) p[part.type] = part.value;
  const wallAsUtc = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second);
  return wallAsUtc - ms;
}

// NZ local midnight of an ISO date, converted to UTC (handles NZST/NZDT).
function nzMidnightToUtc(dateStr) {
  const off = zonedOffsetMs(Date.parse(`${dateStr}T12:00:00Z`));
  return new Date(Date.parse(`${dateStr}T00:00:00Z`) - off).toISOString();
}

// Decimal degrees -> "S 36° 44′" with hemisphere letter.
function toDMS(coord, posLetter, negLetter) {
  const neg = coord < 0;
  const abs = Math.abs(coord);
  const deg = Math.floor(abs);
  const minutes = Math.round((abs - deg) * 60);
  return `${neg ? negLetter : posLetter} ${deg}° ${minutes}′`;
}

// add n whole days to an ISO "YYYY-MM-DD", returning an ISO "YYYY-MM-DD".
function addDays(dateStr, n) {
  return new Date(Date.parse(`${dateStr}T00:00:00Z`) + n * 86400000)
    .toISOString().slice(0, 10);
}

// ISO "YYYY-MM-DD" -> "21 Dec 2025".
function formatDate(iso) {
  if (!iso) return '';
  const [y, m, d] = iso.split('-');
  return `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}`;
}

function Compass({ tilt, azimuth }) {
  const rad = ((azimuth - 90) * Math.PI) / 180; // screen: 0=N up, rotate CW
  const x = 50 + 26 * Math.cos(rad);
  const y = 50 + 26 * Math.sin(rad);
  return (
    <svg viewBox="0 0 100 100" width="120" height="120" className="compass">
      <circle cx="50" cy="50" r="46" fill="none" stroke="#334" strokeWidth="2" />
      {['N', 'E', 'S', 'W'].map((d, i) => {
        const a = (i * 90 - 90) * (Math.PI / 180);
        const lx = 50 + 40 * Math.cos(a);
        const ly = 50 + 40 * Math.sin(a);
        return (
          <text key={d} x={lx} y={ly} textAnchor="middle" dominantBaseline="central"
            className="compass-label">{d}</text>
        );
      })}
      <line x1="50" y1="50" x2={x} y2={y} stroke="#e2431e" strokeWidth="3" />
      <circle cx="50" cy="50" r="3" fill="#e2431e" />
    </svg>
  );
}

const HELP = {
  city: {
    title: 'Location',
    text: 'Choose which city to model. Each location uses its own CAMS solar-radiation dataset and its own latitude/longitude/altitude (read from the file header). The two options are Auckland and Christchurch.',
  },
  tilt: {
    title: 'Panel tilt',
    text: 'Angle of the panel from horizontal, in degrees. 0° = flat on the ground, 90° = vertical. For a fixed NZ roof a tilt of ~25–35° is typical.',
  },
  azimuth: {
    title: 'Panel azimuth',
    text: 'Horizontal direction the panel faces, measured clockwise from north. 0° = north, 90° = east, 180° = south, 270° = west. In the southern hemisphere, north-facing panels (azimuth ≈ 0°) usually capture the most energy.',
  },
  power: {
    title: 'Rated power (kWp)',
    text: 'The system’s STC rated DC power in kilowatt-peak (typical home systems are 3–10 kWp). The model produces 1 kW per 1000 W/m² of plane-of-array irradiance, clipped at this rating.',
  },
  albedo: {
    title: 'Albedo',
    text: 'Ground reflectance, from 0 to 1. It scales how much sunlight the panel receives reflected off the ground. Common values: ~0.2 for grass/soil, 0.8+ for snow.',
  },
  model: {
    title: 'Transposition model',
    text: 'Which pvlib model converts horizontal irradiance (GHI/DHI/DNI) onto the tilted panel plane. Perez is the most accurate and is the default; Hay–Davies is a simpler alternative; Isotropic assumes a uniform diffuse sky.',
  },
  inverter: {
    title: 'Inverter efficiency',
    text: 'Fractional AC/DC conversion efficiency. Modern string inverters are typically 95–98% (0.95–0.98), so the default is 0.95. AC power = DC power × efficiency.',
  },
  start: {
    title: 'Start date',
    text: 'First day of the simulation window (a local NZ date, sent to the API as UTC). The dataset spans 2020-01-01 to 2025-12-31.',
  },
  duration: {
    title: 'Duration (days)',
    text: 'How many consecutive days to simulate, from 1 to 31. The charts show every 15-minute interval in the window.',
  },
  year: {
    title: 'Year',
    text: 'Select a full calendar year (2020-2025) to aggregate the 15-minute output over. The model runs the whole NZ calendar year in local time.',
  },
  agg: {
    title: 'Aggregation',
    text: 'Aggregate the annual PV output either by calendar month (12 bars) or by ISO week (~52 bars). Both sum to the same annual total shown in the summary cards.',
  },
  aggYear: {
    title: 'Yearly output chart',
    text: 'Bars show the total AC energy (kWh) produced in each month or ISO week of the selected year. The summary cards above give the full-year totals; the table below lists every bucket with its share of the annual output.',
  },
  stab: {
    title: 'Year-over-year stability',
    text: 'Plots the total annual output (real, with clouds) and the no-cloud reference for every calendar year in the dataset. The table lists each year, and the metrics on the right quantify how stable output is year over year (coefficient of variation and the count of year-over-year changes of at least 5%).',
  },
  price: {
    title: 'Electricity price',
    text: 'Used only to value the solar you can\'t use (wasted). Leave blank to use the bill\'s effective rate (~$0.239/kWh in this dataset). Your actual bill already reflects the real per-hour cost, so savings are computed from the dollars column.',
  },
  moneyEnergy: {
    title: 'Monthly energy',
    text: 'For each month: total consumption (red), how much was drawn from the grid (blue), solar used on-site (green), and solar that couldn\'t be used and was wasted (light green).',
  },
  moneyCost: {
    title: 'Monthly cost',
    text: 'The red bar is what the month\'s electricity would have cost with no solar; the green bar is the actual cost with solar. The gap between them is what you saved.',
  },
  cloudDaily: {
    title: 'Cloud index (daily)',
    text: 'Kc = GHI ÷ clear-sky GHI for each daylight interval. 1.0 = fully clear sky; lower = more cloud. Night periods are left blank (NaN) because the ratio is undefined without sunlight, so the line is only drawn during the day. The dashed line at 1.0 marks a clear sky.',
  },
  cloudYear: {
    title: 'Cloud index (seasonal)',
    text: 'The mean cloud index over each month or ISO week, using the same buckets as the output chart. Only daylight intervals are averaged — night is excluded (blank), because including night zeros would drag the value down. It reveals seasonal cloudiness.',
  },
  chartPower: {
    title: 'PV output chart',
    text: 'Plots inverter AC power (watts) at each 15-minute step. The curve follows sunlight through the day and drops to zero at night. The dashed green line shows what the panel would produce with no clouds (clear-sky reference) and can be toggled off. A mean line plus the period’s Total and no-cloud energy are shown on the chart.',
  },
  chartIrr: {
    title: 'Irradiance chart',
    text: 'Shows the radiation components the model uses: GHI (global horizontal), DNI (direct normal) and POA (plane-of-array — the irradiance actually hitting the tilted panel). POA is what drives power output.',
  },
  chartGeo: {
    title: 'Sun geometry chart',
    text: 'Sun elevation is the sun’s height above the horizon; angle of incidence (AOI) is the angle between the sun’s rays and the panel’s perpendicular. A lower AOI (sun closer to straight-on) gives more direct power.',
  },
};

function Help({ title, text }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="help-wrap">
      <span
        className="help"
        role="button"
        tabIndex={0}
        aria-label={`Help: ${title}`}
        onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen((o) => !o); }
        }}
      >?</span>
      {open && (
        <div className="help-popup" onClick={(e) => e.stopPropagation()}>
          <button className="help-close" onClick={() => setOpen(false)} aria-label="Close">✕</button>
          <h4>{title}</h4>
          <p>{text}</p>
        </div>
      )}
    </span>
  );
}

function Field({ label, help, children }) {
  return (
    <div className="field">
      <span className="field-label">
        {label}
        {help && <Help title={help.title} text={help.text} />}
      </span>
      {children}
    </div>
  );
}

function ChartHead({ title, help }) {
  return (
    <div className="chart-head">
      <h2>{title}</h2>
      <Help title={help.title} text={help.text} />
    </div>
  );
}

// Custom legend so the transparent "Solar wasted" bar still shows a light-green
// swatch (recharts uses the bar fill for the legend icon, which is transparent).
function EnergyLegend({ payload }) {
  const colors = {
    'Consumption': '#e2431e',
    'Grid import': '#1e88e5',
    'Solar used': '#4caf50',
    'Solar wasted': '#9ccc65',
  };
  return (
    <div className="chart-legend">
      {payload.map((entry, i) => (
        <span key={i} className="legend-item">
          <span className="legend-swatch" style={{ background: colors[entry.value] || entry.color }} />
          {entry.value}
        </span>
      ))}
    </div>
  );
}

// Custom tooltip so the transparent "Solar wasted" bar's value is still visible
// (recharts renders the value text in the bar fill color, which is transparent).
function MoneyEnergyTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const colors = {
    'Consumption': '#e2431e',
    'Grid import': '#1e88e5',
    'Solar used': '#4caf50',
    'Solar wasted': '#9ccc65',
  };
  return (
    <div style={{ background: '#fff', border: '1px solid #ccc', borderRadius: 6, padding: '8px 10px', fontSize: 13, boxShadow: '0 2px 6px rgba(0,0,0,0.15)' }}>
      <div style={{ fontWeight: 600, marginBottom: 4, color: '#222' }}>{label}</div>
      {payload.map((entry) => (
        <div key={entry.dataKey} style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#222', lineHeight: 1.6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: colors[entry.name] || entry.color || entry.stroke, flex: '0 0 auto' }} />
          <span>{entry.name}:</span>
          <b>{Math.round(Number(entry.value))} kWh</b>
        </div>
      ))}
    </div>
  );
}

// Custom legend for the cost chart, matching the energy chart's legend style.
function CostLegend({ payload }) {
  const colors = {
    'Without solar ($)': '#fbc02d',
    'With solar ($)': '#4caf50',
  };
  return (
    <div className="chart-legend">
      {payload.map((entry, i) => (
        <span key={i} className="legend-item">
          <span className="legend-swatch" style={{ background: colors[entry.value] || entry.color }} />
          {entry.value}
        </span>
      ))}
    </div>
  );
}

// Custom tooltip for the cost chart, matching the energy chart's tooltip style.
function MoneyCostTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const colors = {
    'Without solar ($)': '#fbc02d',
    'With solar ($)': '#4caf50',
  };
  return (
    <div style={{ background: '#fff', border: '1px solid #ccc', borderRadius: 6, padding: '8px 10px', fontSize: 13, boxShadow: '0 2px 6px rgba(0,0,0,0.15)' }}>
      <div style={{ fontWeight: 600, marginBottom: 4, color: '#222' }}>{label}</div>
      {payload.map((entry) => (
        <div key={entry.dataKey} style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#222', lineHeight: 1.6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: colors[entry.name] || entry.color || entry.stroke, flex: '0 0 auto' }} />
          <span>{entry.name}:</span>
          <b>${Math.round(Number(entry.value))}</b>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [locations, setLocations] = useState([]);
  const [location, setLocation] = useState('auckland');
  const [panel, setPanel] = useState({
    tilt: 25, azimuth: 0, rated_power_kwp: 5.0, albedo: 0.2,
    transposition_model: 'perez', inverter_efficiency: 0.95,
  });
  const [startDate, setStartDate] = useState('2025-08-18');
  const [days, setDays] = useState(1);
  const [showClear, setShowClear] = useState(true);

  const [activeTab, setActiveTab] = useState('daily');
  const [year, setYear] = useState('2025');
  const [aggPeriod, setAggPeriod] = useState('week');
  const [showClearAgg, setShowClearAgg] = useState(true);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [aggResult, setAggResult] = useState(null);
  const [aggLoading, setAggLoading] = useState(false);
  const [aggError, setAggError] = useState(null);

  const [stabResult, setStabResult] = useState(null);
  const [stabLoading, setStabLoading] = useState(false);
  const [stabError, setStabError] = useState(null);

  const [moneyResult, setMoneyResult] = useState(null);
  const [moneyLoading, setMoneyLoading] = useState(false);
  const [moneyError, setMoneyError] = useState(null);

  const [dqResult, setDqResult] = useState(null);
  const [dqLoading, setDqLoading] = useState(false);
  const [dqError, setDqError] = useState(null);

  useEffect(() => {
    getLocations().then(setLocations).catch((e) => setError(e.message));
  }, []);

  const set = (key) => (e) => {
    const v = e.target.value;
    setPanel((p) => ({ ...p, [key]: e.target.type === 'number' ? Number(v) : v }));
  };

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    const start = nzMidnightToUtc(startDate); // NZ-local 00:00 of the picked date
    const end = new Date(Date.parse(start) + days * 86400000).toISOString();
    try {
      const data = await simulate({ location, start, end, panel });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [location, panel, startDate, days]);

  useEffect(() => { if (activeTab === 'daily') run(); }, [run, activeTab]);

  const runYear = useCallback(async () => {
    setAggLoading(true);
    setAggError(null);
    try {
      const data = await aggregate({
        location, year: Number(year), period: aggPeriod, panel,
      });
      setAggResult(data);
    } catch (e) {
      setAggError(e.message);
    } finally {
      setAggLoading(false);
    }
  }, [location, panel, year, aggPeriod]);

  useEffect(() => { if (activeTab === 'year') runYear(); }, [runYear, activeTab]);

  const runStability = useCallback(async () => {
    setStabLoading(true);
    setStabError(null);
    try {
      const data = await stability({ location, panel });
      setStabResult(data);
    } catch (e) {
      setStabError(e.message);
    } finally {
      setStabLoading(false);
    }
  }, [location, panel]);

  useEffect(() => { if (activeTab === 'stability') runStability(); }, [runStability, activeTab]);

  const runMoney = useCallback(async () => {
    setMoneyLoading(true);
    setMoneyError(null);
    try {
      const data = await money({ location: 'christchurch', panel });
      setMoneyResult(data);
    } catch (e) {
      setMoneyError(e.message);
    } finally {
      setMoneyLoading(false);
    }
  }, [panel]);

  useEffect(() => { if (activeTab === 'money') runMoney(); }, [runMoney, activeTab]);

  const runDataQuality = useCallback(async () => {
    setDqLoading(true);
    setDqError(null);
    try {
      const data = await dataQuality(location);
      setDqResult(data);
    } catch (e) {
      setDqError(e.message);
    } finally {
      setDqLoading(false);
    }
  }, [location]);

  useEffect(() => { if (activeTab === 'dataq') runDataQuality(); }, [runDataQuality, activeTab]);

  // Tab switching: "My money" is locked to Christchurch; leaving it reverts to
  // the default location (Auckland). Other tabs keep whatever the user picked.
  const switchTab = (tab) => {
    if (tab === 'money') {
      setLocation('christchurch');
    } else if (activeTab === 'money') {
      setLocation('auckland');
    }
    setActiveTab(tab);
  };

  const meta = locations.find((l) => l.key === location);
  const timeseries = useMemo(() => result?.timeseries ?? [], [result]);
  const summary = result?.summary ?? null;
  const periodLabel = days === 1
    ? formatDate(startDate)
    : `${formatDate(startDate)}–${formatDate(addDays(startDate, days - 1))}`;
  const aggSummary = aggResult?.summary ?? null;
  const aggBuckets = aggResult?.buckets ?? [];
  const aggTitle = `${aggPeriod === 'month' ? 'Monthly' : 'Weekly'} output (kWh), ${year}`;
  const stabYears = stabResult?.years ?? [];
  const stabMetrics = stabResult?.metrics ?? null;
  const moneyTotals = moneyResult?.totals ?? null;
  const busy = activeTab === 'daily' ? loading
    : activeTab === 'year' ? aggLoading
    : activeTab === 'stability' ? stabLoading
    : activeTab === 'dataq' ? dqLoading
    : moneyLoading;
  const moneyMonthly = useMemo(() => (moneyResult?.monthly ?? []).map((m) => ({
    ...m,
    cost_without: m.cost_$,
    cost_with: +(m.cost_$ - m.savings_$).toFixed(2),
    wasted_pct: m.solar_kwh > 0 ? Math.round((m.excess_kwh / m.solar_kwh) * 100) : 0,
  })), [moneyResult]);
  const moneyTotalRow = useMemo(() => moneyMonthly.reduce((a, m) => {
    a.consumption_kwh += m.consumption_kwh;
    a.solar_kwh += m.solar_kwh;
    a.self_consumed_kwh += m.self_consumed_kwh;
    a.excess_kwh += m.excess_kwh;
    a.grid_kwh += m.grid_kwh;
    a.cost_$ += m.cost_$;
    a.savings_$ += m.savings_$;
    a.waste_$ += m.waste_$;
    return a;
  }, { consumption_kwh: 0, solar_kwh: 0, self_consumed_kwh: 0, excess_kwh: 0,
       grid_kwh: 0, cost_$: 0, savings_$: 0, waste_$: 0 }), [moneyMonthly]);
  return (
    <div className="app">
      <header>
        <div className="header-inner">
          <div>
            <h1>☀️ NZ Solar PV Model</h1>
            <p className="subtitle">Idealized PV output from CAMS radiation · switchable location</p>
          </div>
          <div className="header-actions">
            <a className="hdr-link" href="https://github.com/lunar-me/nz-solar-model"
              target="_blank" rel="noopener noreferrer"
              title="View source on GitHub" aria-label="View source on GitHub">
              <FaGithub size={15} />
              <span>GitHub</span>
            </a>
            <a className="hdr-link" href="https://luna-lab.mywire.org/"
              target="_blank" rel="noopener noreferrer"
              title="Author's site — Luna Lab" aria-label="Author's site — Luna Lab">
              <GiMoon size={15} />
              <span>Luna Lab</span>
            </a>
            <a className="hdr-link" href="/paper.html" target="_blank" rel="noopener noreferrer"
              title="Read the paper" aria-label="Read the paper">
              <FaRegFileAlt size={15} />
              <span>Paper</span>
            </a>
          </div>
        </div>
      </header>
      <div className="tabs">
        <button className={activeTab === 'daily' ? 'tab active' : 'tab'} onClick={() => switchTab('daily')}>Daily</button>
        <button className={activeTab === 'year' ? 'tab active' : 'tab'} onClick={() => switchTab('year')}>Year</button>
        <button className={activeTab === 'stability' ? 'tab active' : 'tab'} onClick={() => switchTab('stability')}>Stability</button>
        <button className={activeTab === 'money' ? 'tab active' : 'tab'} onClick={() => switchTab('money')}>$ My money</button>
        <button className={activeTab === 'dataq' ? 'tab active' : 'tab'} onClick={() => switchTab('dataq')}>Data quality</button>
      </div>
      <div className="layout">
        <aside className="controls">
          <fieldset className="controls-field" disabled={busy}>
          <section>
            <h2>Location</h2>
            <Field label="City" help={HELP.city}>
              <select
                value={activeTab === 'money' ? 'christchurch' : location}
                onChange={(e) => setLocation(e.target.value)}
                disabled={activeTab === 'money'}
              >
                {locations.map((l) => (
                  <option key={l.key} value={l.key}>{l.name}</option>
                ))}
              </select>
            </Field>
            {meta && (
              <p className="meta">
                {meta.name}, {meta.region} · lat {toDMS(meta.metadata.latitude, 'N', 'S')} ·
                lon {toDMS(meta.metadata.longitude, 'E', 'W')} · alt {meta.metadata.altitude} m
              </p>
            )}
          </section>

          {activeTab !== 'dataq' && (
          <section>
            <h2>Panel</h2>
            <Field label={`Tilt ${panel.tilt}°`} help={HELP.tilt}>
              <input type="range" min="0" max="90" value={panel.tilt}
                onChange={set('tilt')} />
            </Field>
            <Field label={`Azimuth ${panel.azimuth}° (0=N 90=E 180=S 270=W)`} help={HELP.azimuth}>
              <input type="range" min="0" max="360" value={panel.azimuth}
                onChange={set('azimuth')} />
            </Field>
            <div className="compass-wrap">
              <Compass tilt={panel.tilt} azimuth={panel.azimuth} />
            </div>
            <Field label="Rated power (kWp)" help={HELP.power}>
              <input type="number" min="1" step="1" value={panel.rated_power_kwp}
                onChange={set('rated_power_kwp')} />
            </Field>
            <Field label="Albedo" help={HELP.albedo}>
              <input type="number" min="0" max="1" step="0.05" value={panel.albedo}
                onChange={set('albedo')} />
            </Field>
            <Field label="Transposition model" help={HELP.model}>
              <select value={panel.transposition_model} onChange={set('transposition_model')}>
                {TRANSPOSITION_MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Field>
            <Field label="Inverter efficiency" help={HELP.inverter}>
              <input type="number" min="0.5" max="1" step="0.01"
                value={panel.inverter_efficiency} onChange={set('inverter_efficiency')} />
            </Field>
          </section>
          )}

          {activeTab === 'daily' && (
            <section>
              <h2>Date range</h2>
              <Field label="Start date" help={HELP.start}>
                <input type="date" value={startDate} min="2020-01-01" max="2025-12-31"
                  onChange={(e) => setStartDate(e.target.value)} />
              </Field>
              <Field label="Duration (days)" help={HELP.duration}>
                <input type="number" min="1" max="31" value={days}
                  onChange={(e) => setDays(Math.max(1, Math.min(31, Number(e.target.value))))} />
              </Field>
            </section>
          )}

          {activeTab === 'year' && (
            <section>
              <h2>Year</h2>
              <Field label="Year" help={HELP.year}>
                <select value={year} onChange={(e) => setYear(e.target.value)}>
                  {[2020, 2021, 2022, 2023, 2024, 2025].map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </Field>
              <Field label="Aggregation" help={HELP.agg}>
                <select value={aggPeriod} onChange={(e) => setAggPeriod(e.target.value)}>
                  <option value="month">Monthly</option>
                  <option value="week">Weekly</option>
                </select>
              </Field>
            </section>
          )}

                    </fieldset>

          <button
            onClick={() => (activeTab === 'daily' ? run()
              : activeTab === 'year' ? runYear()
              : activeTab === 'stability' ? runStability()
              : activeTab === 'dataq' ? runDataQuality()
              : runMoney())}
            disabled={activeTab === 'daily' ? loading
              : activeTab === 'year' ? aggLoading
              : activeTab === 'stability' ? stabLoading
              : activeTab === 'dataq' ? dqLoading
              : moneyLoading}
            className="run"
          >
            {(activeTab === 'daily' ? loading
              : activeTab === 'year' ? aggLoading
              : activeTab === 'stability' ? stabLoading
              : activeTab === 'dataq' ? dqLoading
              : moneyLoading) ? 'Running\u2026' : 'Run'}
          </button>
        </aside>

        <main className="content">
          {activeTab === 'daily' && (
            <>
          {error && <div className="error">{error}</div>}

          {!result && !error && (
            <p className="hint">Loading... please wait</p>
          )}

          {result && (
            <>
              <section className="report">
                <div className="report-head">
                  <h2>Energy summary</h2>
                  <span className="report-period">
                    {periodLabel}{days > 1 ? ` · ${days} days` : ''}
                  </span>
                </div>
                <div className="report-cards">
                  <div className="rcard"><span>Installed PV</span><b>{panel.rated_power_kwp} kWp</b></div>
                  <div className="rcard"><span>Total</span><b>{summary.total_energy_kwh} kWh</b></div>
                  <div className="rcard"><span>No cloud</span><b>{summary.total_energy_clear_kwh} kWh</b></div>
                  <div className="rcard"><span>Peak</span><b>{summary.peak_power_kw} kW</b></div>
                  <div className="rcard"><span>Yield</span><b>{summary.specific_yield_kwh_per_kwp} kWh/kWp</b></div>
                  <div className="rcard"><span>Mean AC</span><b>{Math.round(summary.mean_ac_power_w)} W</b></div>
                </div>
              </section>

              <section className="chart-block">
                <ChartHead title="PV output (AC power, W)" help={HELP.chartPower} />
                <div className="chart-controls">
                  <label className="toggle">
                    <input type="checkbox" checked={showClear}
                      onChange={(e) => setShowClear(e.target.checked)} />
                    Show no-cloud line
                  </label>
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height={480}>
                    <ComposedChart data={timeseries}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp_local" tickFormatter={formatTick} minTickGap={40} />
                      <YAxis tickFormatter={(v) => Math.round(v)} />
                      <Tooltip
                        labelFormatter={formatLocalFull}
                        formatter={(value, name) => [Math.round(Number(value)), name]}
                      />
                      <Legend />
                      <Area type="monotone" dataKey="ac_power" name="AC power (W, real clouds)"
                            fill="#4caf50" fillOpacity={0.25} stroke="#2e7d32" />
                      {showClear && (
                        <Line type="monotone" dataKey="ac_power_clear" name="AC power (W, clear sky)"
                          stroke="#9ccc65" strokeDasharray="6 4" dot={false} strokeWidth={1.5} />
                      )}
                      <ReferenceLine
                        y={summary.mean_ac_power_w}
                        stroke="#ffb020"
                        label={{
                          value: `mean ${Math.round(summary.mean_ac_power_w)} W`,
                          position: 'insideTopRight', fill: '#ffb020', fontSize: 12,
                        }}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                  <div className="chart-badges">
                    <div className="chart-badge">Total: {summary.total_energy_kwh} kWh</div>
                    <div className="chart-badge chart-badge-clear">
                      No cloud: {summary.total_energy_clear_kwh} kWh
                    </div>
                  </div>
                </div>
              </section>

              <section className="chart-block">
                <ChartHead title="Cloud index (GHI / GHI_clear)" help={HELP.cloudDaily} />
                <div className="chart">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={timeseries}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp_local" tickFormatter={formatTick} minTickGap={40} />
                      <YAxis domain={[0, 1.2]} />
                      <Tooltip labelFormatter={formatLocalFull} formatter={(v) => [v, 'Cloud index']} labelStyle={{ color: '#222' }} />
                      <ReferenceLine y={1} stroke="#ffb020" strokeDasharray="4 3"
                        label={{ value: 'clear', position: 'insideTopRight', fill: '#ffb020', fontSize: 11 }} />
                      <Line type="monotone" dataKey="cloud_index" stroke="#1e88e5" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="cloud-note">Cloud index = GHI ÷ clear-sky GHI. 1.0 = fully clear; lower = more cloud.</p>
              </section>

              <section className="chart-block">
                <ChartHead title="Irradiance (W/m²)" help={HELP.chartIrr} />
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={timeseries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp_local" tickFormatter={formatTick} minTickGap={40} />
                    <YAxis />
                    <Tooltip labelFormatter={formatLocalFull} />
                    <Legend />
                    <Line type="monotone" dataKey="ghi" name="GHI (global horizontal)" stroke="#1e88e5" dot={false} />
                    <Line type="monotone" dataKey="dni" name="DNI (direct normal)" stroke="#f4511e" dot={false} />
                    <Line type="monotone" dataKey="poa_global" name="POA (plane-of-array)" stroke="#6a1b9a" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </section>

              <section className="chart-block">
                <ChartHead title="Sun geometry &amp; angle of incidence (°)" help={HELP.chartGeo} />
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={timeseries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp_local" tickFormatter={formatTick} minTickGap={40} />
                    <YAxis domain={[0, 90]} />
                    <Tooltip labelFormatter={formatLocalFull} />
                    <Legend />
                    <Line dataKey="sun_elevation_deg" name="Sun elevation" stroke="#43a047" dot={false} />
                    <Line dataKey="aoi_deg" name="Angle of incidence" stroke="#fb8c00" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </section>
            </>
          )}
            </>
          )}

          {activeTab === 'year' && (
            <>
              {aggError && <div className="error">{aggError}</div>}
              {!aggResult && !aggError && (
                <p className="hint">Loading... please wait</p>
              )}
              {aggResult && (
                <>
                  <section className="report">
                    <div className="report-head">
                      <h2>Annual summary</h2>
                      <span className="report-period">{year}</span>
                    </div>
                    <div className="report-cards">
                      <div className="rcard"><span>Installed PV</span><b>{panel.rated_power_kwp} kWp</b></div>
                      <div className="rcard"><span>Total</span><b>{Math.round(aggSummary.total_energy_kwh)} kWh</b></div>
                      <div className="rcard"><span>No cloud</span><b>{Math.round(aggSummary.total_energy_clear_kwh)} kWh</b></div>
                      <div className="rcard"><span>Peak</span><b>{aggSummary.peak_power_kw} kW</b></div>
                      <div className="rcard"><span>Yield</span><b>{Math.round(aggSummary.specific_yield_kwh_per_kwp)} kWh/kWp</b></div>
                      <div className="rcard"><span>Mean AC</span><b>{Math.round(aggSummary.mean_ac_power_w)} W</b></div>
                    </div>
                  </section>

                  <section className="chart-block">
                    <ChartHead title={aggTitle} help={HELP.aggYear} />
                    <div className="chart-controls">
                      <label className="toggle">
                        <input type="checkbox" checked={showClearAgg}
                          onChange={(e) => setShowClearAgg(e.target.checked)} />
                        Show no-cloud top-up
                      </label>
                    </div>
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={380}>
                        <BarChart data={aggBuckets} margin={{ top: 14, right: 12, bottom: 4, left: 4 }}>
                          <CartesianGrid vertical={false} strokeDasharray="3 3" />
                          <XAxis dataKey="label" interval={aggPeriod === 'week' ? 4 : 0} tickFormatter={(v) => {
                            if (aggPeriod === 'week') {
                              const b = aggBuckets.find((x) => x.label === v);
                              return b && b.week_start ? formatDate(b.week_start) : v;
                            }
                            return v.slice(0, 3);
                          }} />
                          <YAxis tickFormatter={(v) => Math.round(v)} />
                          <Tooltip
                            formatter={(value, name) => [`${value} kWh`, name]}
                            labelFormatter={(label) => {
                              if (aggPeriod === 'week') {
                                const b = aggBuckets.find((x) => x.label === label);
                                return b && b.week_start ? `WC ${formatDate(b.week_start)}` : label;
                              }
                              return label;
                            }}
                            labelStyle={{ color: '#222' }}
                          />
                          <Legend />
                          <Bar dataKey="energy_kwh" name="Output (real clouds)" stackId="a" fill="#4caf50" radius={[0, 0, 0, 0]}>
                            {aggPeriod === 'month' && (
                              <LabelList dataKey="energy_kwh" position="top" formatter={(v) => `${Math.round(v)} kWh`} />
                            )}
                          </Bar>
                          {showClearAgg && (
                            <Bar dataKey="no_cloud_extra" name="No cloud (top-up)" stackId="a"
                              fill="#9ccc65" stroke="#6aa84f" strokeDasharray="4 3" fillOpacity={0.55} radius={[3, 3, 0, 0]}>
                              {aggPeriod === 'month' && (
                                <LabelList dataKey="energy_clear_kwh" position="top" formatter={(v) => `${Math.round(v)} kWh`} />
                              )}
                            </Bar>
                          )}
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <div className="stab-row">
                    <section className="chart-block stab-table-block">
                      <div className="chart-head"><h2>{aggPeriod === 'month' ? 'Monthly' : 'Weekly'} table</h2></div>
                      <div className="agg-table-wrap">
                        <table className="agg-table">
                          <thead>
                            <tr>
                              <th>Period</th>
                              {aggPeriod === 'week' && <th>WC Date</th>}
                              <th>Energy (kWh)</th>
                              <th>No cloud (kWh)</th>
                              <th>Share</th>
                            </tr>
                          </thead>
                          <tbody>
                            {aggBuckets.map((b) => (
                              <tr key={b.key}>
                                <td>{b.label}</td>
                                {aggPeriod === 'week' && <td>{formatDate(b.week_start)}</td>}
                                <td>{Math.round(b.energy_kwh * 10) / 10}</td>
                                <td>{Math.round(b.energy_clear_kwh * 10) / 10}</td>
                                <td>{Math.round(b.share * 100)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="chart-block cloud-block">
                      <ChartHead title="Cloud index (GHI / GHI_clear)" help={HELP.cloudYear} />
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={aggBuckets}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="label" interval={aggPeriod === 'week' ? 4 : 0}
                              tickFormatter={(v) => (aggPeriod === 'month'
                                ? v.slice(0, 3)
                                : (aggBuckets.find((x) => x.label === v)?.week_start
                                  ? formatDate(aggBuckets.find((x) => x.label === v).week_start)
                                  : v))} />
                            <YAxis domain={[0, 1.2]} />
                            <Tooltip formatter={(v) => [v, 'Cloud index']} labelStyle={{ color: '#222' }} />
                            <ReferenceLine y={1} stroke="#ffb020" strokeDasharray="4 3"
                              label={{ value: 'clear', position: 'insideTopRight', fill: '#ffb020', fontSize: 11 }} />
                            <Line type="monotone" dataKey="cloud_index" stroke="#1e88e5" dot={false} strokeWidth={2} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="cloud-note">
                        Cloud index = GHI ÷ clear-sky GHI. 1.0 = fully clear sky; lower = more cloud.
                        The chart shows the mean over each {aggPeriod} of the selected year, using the
                        same {aggPeriod === 'month' ? 'months' : 'weeks'} as the output chart.
                      </p>
                    </section>
                  </div>
                </>
              )}
            </>
          )}

          {activeTab === 'stability' && (
            <>
              {stabError && <div className="error">{stabError}</div>}
              {!stabResult && !stabError && (
                <p className="hint">Loading... please wait</p>
              )}
              {stabResult && (
                <>
                  <section className="chart-block">
                    <ChartHead title="Year-over-year PV output (kWh)" help={HELP.stab} />
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={360}>
                        <LineChart data={stabYears}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="year" />
                          <YAxis tickFormatter={(v) => Math.round(v)} />
                          <Tooltip formatter={(value, name) => [`${value} kWh`, name]} labelStyle={{ color: '#222' }} />
                          <Legend />
                          <Line type="monotone" dataKey="total_energy_kwh" name="Total output (kWh)" stroke="#4caf50" strokeWidth={2} dot={{ r: 5 }} />
                          <Line type="monotone" dataKey="total_energy_clear_kwh" name="No cloud (kWh)" stroke="#9ccc65" strokeDasharray="6 4" strokeWidth={2} dot={{ r: 5 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <div className="stab-row">
                    <section className="chart-block stab-table-block">
                      <div className="chart-head"><h2>Annual totals</h2></div>
                      <div className="agg-table-wrap">
                        <table className="agg-table stab-table">
                          <thead>
                            <tr>
                              <th>Year</th>
                              <th>Total (kWh)</th>
                              <th>No cloud (kWh)</th>
                              <th>Loss (kWh)</th>
                              <th>Loss %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {stabYears.map((y) => (
                              <tr key={y.year}>
                                <td>{y.year}</td>
                                <td>{Math.round(y.total_energy_kwh)}</td>
                                <td>{Math.round(y.total_energy_clear_kwh)}</td>
                                <td>{Math.round(y.cloud_loss_kwh)}</td>
                                <td>{y.cloud_loss_pct}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>

                    <section className="stab-metrics">
                      <h3>Year-over-year stability</h3>
                      <div className="stab-metric"><span>Coefficient of variation</span><b>{stabMetrics.cv_pct}%</b></div>
                      <div className="stab-metric"><span>Average output</span><b>{stabMetrics.mean_kwh} kWh</b></div>
                      <div className="stab-metric"><span>Min / Max</span><b>{stabMetrics.min_kwh} ({stabMetrics.min_year}) / {stabMetrics.max_kwh} ({stabMetrics.max_year}) kWh</b></div>
                      <div className="stab-metric"><span>Range</span><b>{stabMetrics.range_kwh} kWh</b></div>
                      <div className="stab-metric"><span>Variations (≥5%)</span><b>{stabMetrics.variations} of {stabMetrics.transitions}</b></div>
                      <ul className="stab-yoy">
                        {stabMetrics.yoy.map((c) => (
                          <li key={`${c.from}-${c.to}`}>
                            {c.from} → {c.to}: {c.change_pct > 0 ? '+' : ''}{c.change_pct}%
                          </li>
                        ))}
                      </ul>
                    </section>
                  </div>
                </>
              )}
            </>
          )}

          {activeTab === 'money' && (
            <>
              {moneyError && <div className="error">{moneyError}</div>}
              {!moneyResult && !moneyError && (<p className="hint">Loading... please wait</p>)}
              {moneyResult && (
                <>
                  <div className="explain">
                    This tab models my hourly Christchurch electricity consumption
                    (18 Aug 2025 – 17 Aug 2026) against the hourly solar output the panel
                    on the left could produce. Each hour, the solar I'd use on-site is
                    subtracted from what I'd pay; any solar I couldn't use is counted as
                    wasted. Savings and the value of wasted solar are computed
                    hour-by-hour using each hour's actual bill (the dollars column) — no
                    average electricity rate is assumed. All amounts are in NZ dollars.
                  </div>

                  <section className="report">
                    <div className="report-head">
                      <h2>Solar savings</h2>
                      <span className="report-period">18 Aug 2025 – 17 Aug 2026 · Christchurch</span>
                    </div>
                    <div className="report-cards">
                      <div className="rcard"><span>Installed PV</span><b>{panel.rated_power_kwp} kWp</b></div>
                      <div className="rcard money-highlight"><span>Savings</span><b>${moneyTotals.savings_$} ({moneyTotals.savings_pct}%)</b></div>
                      <div className="rcard"><span>Bill without solar</span><b>${moneyTotals.cost_without_solar_$}</b></div>
                      <div className="rcard"><span>Bill with solar</span><b>${moneyTotals.cost_with_solar_$}</b></div>
                      <div className="rcard"><span>Wasted solar value</span><b>${moneyTotals.wasted_value_$}</b></div>
                      <div className="rcard"><span>Self-consumed</span><b>{moneyTotals.self_consumed_kwh} kWh ({moneyTotals.self_consumption_pct}%)</b></div>
                      <div className="rcard"><span>Grid import</span><b>{moneyTotals.grid_import_kwh} kWh</b></div>
                    </div>
                  </section>

                  <section className="chart-block">
                    <ChartHead title="Monthly energy (kWh)" help={HELP.moneyEnergy} />
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={340}>
                        <BarChart data={moneyMonthly} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                          <CartesianGrid vertical={false} strokeDasharray="3 3" />
                          <XAxis dataKey="label" interval={0} tickFormatter={(v) => v.slice(0, 3)} />
                          <YAxis />
                          <Tooltip content={<MoneyEnergyTooltip />} />
                          <Legend content={EnergyLegend} />
                          <Bar dataKey="consumption_kwh" name="Consumption" fill="#e2431e" radius={[2, 2, 0, 0]} />
                          <Bar dataKey="grid_kwh" name="Grid import" fill="#1e88e5" radius={[2, 2, 0, 0]} />
                          <Bar dataKey="self_consumed_kwh" name="Solar used" fill="#4caf50" radius={[2, 2, 0, 0]} />
                          <Bar dataKey="excess_kwh" name="Solar wasted" fill="transparent" stroke="#4caf50" strokeWidth={2} radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="chart-block">
                    <ChartHead title="Monthly cost (NZD)" help={HELP.moneyCost} />
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={moneyMonthly} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                          <CartesianGrid vertical={false} strokeDasharray="3 3" />
                          <XAxis dataKey="label" interval={0} tickFormatter={(v) => v.slice(0, 3)} />
                          <YAxis />
                          <Tooltip content={<MoneyCostTooltip />} />
                          <Legend content={CostLegend} />
                          <Bar dataKey="cost_without" name="Without solar ($)" fill="#fbc02d" radius={[2, 2, 0, 0]}>
                            <LabelList dataKey="cost_without" position="top" formatter={(v) => `$${Math.round(v)}`} />
                          </Bar>
                          <Bar dataKey="cost_with" name="With solar ($)" fill="#4caf50" radius={[2, 2, 0, 0]}>
                            <LabelList dataKey="cost_with" position="top" formatter={(v) => `$${Math.round(v)}`} />
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </section>

                  <section className="chart-block">
                    <div className="chart-head"><h2>Monthly detail</h2></div>
                    <div className="agg-table-wrap">
                      <table className="agg-table money-table">
                        <thead>
                          <tr>
                            <th>Month</th><th>Use (kWh)</th><th>Solar (kWh)</th><th>Used (kWh)</th>
                            <th>Wasted (kWh)</th><th>Wasted %</th><th>Grid (kWh)</th><th>Cost ($)</th><th>Saved ($)</th><th>Wasted ($)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {moneyMonthly.map((m) => (
                            <tr key={m.month}>
                              <td>{m.label}</td>
                              <td>{Math.round(m.consumption_kwh)}</td>
                              <td>{Math.round(m.solar_kwh)}</td>
                              <td>{Math.round(m.self_consumed_kwh)}</td>
                              <td>{Math.round(m.excess_kwh)}</td>
                              <td>{m.wasted_pct}%</td>
                              <td>{Math.round(m.grid_kwh)}</td>
                              <td>{Math.round(m.cost_$)}</td>
                              <td>{Math.round(m.savings_$)}</td>
                              <td>{Math.round(m.waste_$)}</td>
                            </tr>
                          ))}
                          <tr className="agg-total">
                            <td>Total</td>
                            <td>{Math.round(moneyTotalRow.consumption_kwh)}</td>
                            <td>{Math.round(moneyTotalRow.solar_kwh)}</td>
                            <td>{Math.round(moneyTotalRow.self_consumed_kwh)}</td>
                            <td>{Math.round(moneyTotalRow.excess_kwh)}</td>
                            <td>{Math.round((moneyTotalRow.excess_kwh / moneyTotalRow.solar_kwh) * 100)}%</td>
                            <td>{Math.round(moneyTotalRow.grid_kwh)}</td>
                            <td>{Math.round(moneyTotalRow.cost_$)}</td>
                            <td>{Math.round(moneyTotalRow.savings_$)}</td>
                            <td>{Math.round(moneyTotalRow.waste_$)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <div className="money-final">
                    Over the year, solar could save me <b>${moneyTotals.savings_$}</b> ({moneyTotals.savings_pct}%
                    of my bill) — and <b>{moneyTotals.excess_kwh} kWh</b> of solar (
                    {Math.round((moneyTotals.excess_kwh / moneyTotals.solar_kwh) * 100)}% of what I
                    could produce, worth ~${moneyTotals.wasted_value_$}) would be wasted because
                    I couldn't use it the moment it was produced.
                  </div>
                </>
              )}
            </>
          )}

          {activeTab === 'dataq' && (
            <>
              {dqError && <div className="error">{dqError}</div>}
              {!dqResult && !dqError && (<p className="hint">Loading... please wait</p>)}
              {dqResult && (
                <>
                  <div className="explain">
                    This tab reports on the quality of the <b>CAMS Radiation</b> dataset for the
                    selected city (all-sky irradiance, 15-minute intervals). It does <b>not</b> cover
                    the Christchurch electricity-consumption dataset. CAMS supplies a per-interval
                    <b> Reliability</b> value (0–1) = the proportion of reliable data in each 15-minute
                    interval, based on how much the satellite cloud retrieval could be trusted. The
                    report checks time continuity, radiation plausibility (e.g. no negative values,
                    GHI ≈ DHI + DNI·cos(zenith)) and the reliability distribution.
                  </div>

                  <section className="report">
                    <div className="report-head">
                      <h2>Data quality — {meta?.name}</h2>
                      <span className="report-period">{dqResult.span.start.slice(0, 10)} → {dqResult.span.end.slice(0, 10)}</span>
                    </div>
                    <div className="report-cards">
                      <div className="rcard"><span>Rows</span><b>{dqResult.span.rows}</b></div>
                      <div className="rcard"><span>Interval</span><b>{dqResult.span.interval_h} h</b></div>
                      <div className="rcard"><span>Completeness</span><b>{dqResult.time.completeness_pct}%</b></div>
                      <div className="rcard"><span>Duplicates</span><b>{dqResult.time.duplicates}</b></div>
                      <div className="rcard"><span>Missing</span><b>{dqResult.time.missing_intervals}</b></div>
                      <div className="rcard"><span>Low reliability</span><b>{dqResult.reliability.low_pct}%</b></div>
                    </div>
                  </section>

                  <section className="chart-block">
                    <div className="chart-head"><h2>Time integrity</h2></div>
                    <div className="dq-rows">
                      <div className="dq-row"><span>Expected intervals</span><b>{dqResult.time.expected_intervals}</b></div>
                      <div className="dq-row"><span>Actual rows</span><b>{dqResult.time.rows}</b></div>
                      <div className="dq-row"><span>Duplicate timestamps</span><b>{dqResult.time.duplicates}</b></div>
                      <div className="dq-row"><span>Missing intervals</span><b>{dqResult.time.missing_intervals}</b></div>
                      <div className="dq-row"><span>UTC timezone-aware</span><b>{dqResult.time.timezone_utc_aware ? 'yes' : 'no'}</b></div>
                    </div>
                    {dqResult.time.gaps.length > 0 && (
                      <ul className="dq-list">
                        {dqResult.time.gaps.map((g, i) => <li key={i}>Gap of {g.hours}h after {g.after}</li>)}
                      </ul>
                    )}
                  </section>

                  <section className="chart-block">
                    <div className="chart-head"><h2>Radiation checks</h2></div>
                    <table className="agg-table dq-table">
                      <thead><tr><th>Column</th><th>Negative</th><th>Min (W/m²)</th><th>Max (W/m²)</th></tr></thead>
                      <tbody>
                        {Object.entries(dqResult.radiation.negatives).map(([c, cnt]) => {
                          const [mn, mx] = dqResult.radiation.ranges[c];
                          return (<tr key={c}><td>{c}</td><td>{cnt}</td><td>{mn}</td><td>{mx}</td></tr>);
                        })}
                      </tbody>
                    </table>
                    <div className="dq-rows">
                      <div className="dq-row"><span>GHI ≈ DHI + DNI·cos(z), mean residual</span><b>{dqResult.radiation.ghi_conservation.mean_residual} W/m²</b></div>
                      <div className="dq-row"><span>…max |residual|</span><b>{dqResult.radiation.ghi_conservation.max_abs_residual} W/m²</b></div>
                      <div className="dq-row"><span>DHI ≤ GHI violations</span><b>{dqResult.radiation.dhi_le_ghi_violations}</b></div>
                      <div className="dq-row"><span>BHI ≤ GHI violations</span><b>{dqResult.radiation.bhi_le_ghi_violations}</b></div>
                    </div>
                  </section>

                  <section className="chart-block">
                    <div className="chart-head"><h2>Reliability</h2></div>
                    <div className="dq-rows">
                      <div className="dq-row"><span>Min</span><b>{dqResult.reliability.min}</b></div>
                      <div className="dq-row"><span>Median</span><b>{dqResult.reliability.median}</b></div>
                      <div className="dq-row"><span>Intervals with reliability &lt; 1.0</span><b>{dqResult.reliability.below_1} ({dqResult.reliability.low_pct}%)</b></div>
                      <div className="dq-row"><span>Intervals with reliability &lt; 0.5</span><b>{dqResult.reliability.below_0_5}</b></div>
                    </div>
                  </section>

                  {dqResult.checks.length > 0 && (
                    <section className="chart-block">
                      <div className="chart-head"><h2>Findings</h2></div>
                      <ul className="dq-list">
                        {dqResult.checks.map((c, i) => (
                          <li key={i} className={'dq-' + c.level}>{c.level.toUpperCase()}: {c.msg}</li>
                        ))}
                      </ul>
                    </section>
                  )}
                </>
              )}
            </>
          )}
        </main>

      </div>
    </div>
  );
}

