const BASE = '/api';

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export function getLocations() {
  return fetch(`${BASE}/locations`).then(json);
}

export function simulate(payload) {
  return fetch(`${BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(json);
}

export function aggregate(payload) {
  return fetch(`${BASE}/aggregate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(json);
}

export function stability(payload) {
  return fetch(`${BASE}/stability`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(json);
}

export function money(payload) {
  return fetch(`${BASE}/money`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(json);
}

export function dataQuality(location) {
  return fetch(`${BASE}/data-quality?location=${encodeURIComponent(location)}`).then(json);
}

export function getRadiation(location, start, end) {
  const p = new URLSearchParams();
  if (start) p.set('start', start);
  if (end) p.set('end', end);
  return fetch(`${BASE}/radiation/${location}?${p.toString()}`).then(json);
}
