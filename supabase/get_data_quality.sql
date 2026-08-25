-- ============================================================================
-- Data Quality report, computed in PostgreSQL (not in the app).
--
-- Returns every figure the /api/data-quality endpoint used to compute in
-- pandas over the *full* multi-year dataset, as a single JSON object — so the
-- app no longer downloads ~230k rows just to aggregate them.
--
-- Product decision (2026-08): the two solar-zenith-dependent checks are DROPPED.
--   * ghi_conservation      -> {mean_residual: null, max_abs_residual: null}
--   * bhi_le_ghi_violations -> null
-- They were computed with pvlib's apparent_zenith (precise solar geometry),
-- which does not map trivially to SQL; the rest of the report is identical.
--
-- Run this once in the Supabase SQL editor (or as a migration). The app calls it
-- via PostgREST RPC:  POST /rest/v1/rpc/get_data_quality
--     body: {"location":"Auckland","latitude":-36.73,"longitude":174.71,"altitude":51}
-- ============================================================================

CREATE OR REPLACE FUNCTION public.get_data_quality(
    location   text,
    latitude   double precision,
    longitude  double precision,
    altitude   double precision
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_location   text := location;

    _n           bigint;
    _start       timestamptz;
    _end         timestamptz;
    _span_h      double precision;
    _interval_h  double precision;
    _expected    int;
    _duplicates  bigint;
    _missing     int;
    _completeness numeric;

    _neg_ghi  bigint; _neg_dhi  bigint; _neg_dni  bigint;
    _neg_gc   bigint; _neg_dc   bigint; _neg_dnc  bigint;

    _min_ghi double precision; _max_ghi double precision;
    _min_dhi double precision; _max_dhi double precision;
    _min_dni double precision; _max_dni double precision;
    _min_gc  double precision; _max_gc  double precision;
    _min_dc  double precision; _max_dc  double precision;
    _min_dnc double precision; _max_dnc double precision;

    _dhi_le_ghi bigint;

    _rel_min    double precision;
    _rel_median double precision;
    _rel_below1 bigint;
    _rel_below05 bigint;
    _rel_low_pct double precision;

    _gaps       jsonb;
    _checks     jsonb;
    _status     text;
    _has_error  boolean := false;
    _rec        record;
BEGIN
    SELECT count(*), min(start_ts_utc), max(start_ts_utc)
      INTO _n, _start, _end
      FROM cams_radiation cr
     WHERE cr.location = v_location;

    -- Empty location -> a fully "null" report (keeps the JSON shape stable).
    IF _n = 0 THEN
        RETURN jsonb_build_object(
            'span', jsonb_build_object('start', NULL, 'end', NULL,
                                       'interval_h', NULL, 'rows', 0),
            'time', jsonb_build_object('expected_intervals', 0, 'rows', 0,
                                       'duplicates', 0, 'missing_intervals', 0,
                                       'completeness_pct', 100.0,
                                       'gaps', '[]'::jsonb, 'timezone_utc_aware', true),
            'radiation', jsonb_build_object(
                'negatives', jsonb_build_object('ghi',0,'dhi',0,'dni',0,
                                                'ghi_clear',0,'dhi_clear',0,'dni_clear',0),
                'ranges', jsonb_build_object(
                    'ghi', jsonb_build_array(NULL, NULL), 'dhi', jsonb_build_array(NULL, NULL),
                    'dni', jsonb_build_array(NULL, NULL), 'ghi_clear', jsonb_build_array(NULL, NULL),
                    'dhi_clear', jsonb_build_array(NULL, NULL), 'dni_clear', jsonb_build_array(NULL, NULL)),
                'ghi_conservation', jsonb_build_object('mean_residual', NULL, 'max_abs_residual', NULL),
                'dhi_le_ghi_violations', 0,
                'bhi_le_ghi_violations', NULL),
            'reliability', jsonb_build_object('min', NULL, 'median', NULL,
                                              'below_1', 0, 'below_0_5', 0, 'low_pct', 0.0),
            'checks', '[]'::jsonb,
            'status', 'good'
        );
    END IF;

    -- Median interval length (hours) between consecutive timestamps.
    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY d)
      INTO _interval_h
      FROM (
        SELECT EXTRACT(EPOCH FROM (start_ts_utc - LAG(start_ts_utc)
                                   OVER (ORDER BY start_ts_utc))) / 3600.0 AS d
          FROM cams_radiation cr
         WHERE cr.location = v_location
      ) t
     WHERE d IS NOT NULL;

    _span_h := EXTRACT(EPOCH FROM (_end - _start)) / 3600.0;
    _expected := CASE WHEN _interval_h IS NOT NULL AND _interval_h > 0
                      THEN round(_span_h / _interval_h)::int + 1
                      ELSE _n::int END;
    _duplicates := _n - (SELECT count(DISTINCT cr.start_ts_utc)
                           FROM cams_radiation cr WHERE cr.location = v_location);
    _missing := greatest(_expected - _n::int, 0);
    _completeness := round((_n::numeric / NULLIF(_expected, 0)) * 100.0, 2);

    -- Negative-value counts per radiation column (internal names).
    SELECT count(*) FILTER (WHERE cr.ghi < 0),
           count(*) FILTER (WHERE cr.dhi < 0),
           count(*) FILTER (WHERE cr.bni < 0),
           count(*) FILTER (WHERE cr.clear_sky_ghi < 0),
           count(*) FILTER (WHERE cr.clear_sky_dhi < 0),
           count(*) FILTER (WHERE cr.clear_sky_bni < 0)
      INTO _neg_ghi, _neg_dhi, _neg_dni, _neg_gc, _neg_dc, _neg_dnc
      FROM cams_radiation cr WHERE cr.location = v_location;

    -- Per-column min/max (W/m2).
    SELECT min(cr.ghi), max(cr.ghi), min(cr.dhi), max(cr.dhi),
           min(cr.bni), max(cr.bni),
           min(cr.clear_sky_ghi), max(cr.clear_sky_ghi),
           min(cr.clear_sky_dhi), max(cr.clear_sky_dhi),
           min(cr.clear_sky_bni), max(cr.clear_sky_bni)
      INTO _min_ghi, _max_ghi, _min_dhi, _max_dhi,
           _min_dni, _max_dni, _min_gc, _max_gc, _min_dc, _max_dc, _min_dnc, _max_dnc
      FROM cams_radiation cr WHERE cr.location = v_location;

    -- DHI <= GHI plausibility (the only radiation check that needs no geometry).
    SELECT count(*) INTO _dhi_le_ghi
      FROM cams_radiation cr
     WHERE cr.location = v_location AND cr.dhi > cr.ghi + 0.01;

    -- Reliability distribution.
    SELECT min(cr.reliability),
           percentile_cont(0.5) WITHIN GROUP (ORDER BY cr.reliability),
           count(*) FILTER (WHERE cr.reliability < 1.0),
           count(*) FILTER (WHERE cr.reliability < 0.5)
      INTO _rel_min, _rel_median, _rel_below1, _rel_below05
      FROM cams_radiation cr WHERE cr.location = v_location;
    _rel_low_pct := round((_rel_below1::numeric / _n) * 100.0, 2);

    -- Top-5 gaps (intervals more than 1.5x the median).
    SELECT COALESCE(jsonb_agg(
               jsonb_build_object('after', to_char(ts, 'YYYY-MM-DD HH24:MI:SS'),
                                  'hours', round(diff_h, 2))
               ORDER BY diff_h DESC), '[]'::jsonb)
      INTO _gaps
      FROM (
        SELECT start_ts_utc AS ts,
               EXTRACT(EPOCH FROM (start_ts_utc - LAG(start_ts_utc)
                                   OVER (ORDER BY start_ts_utc))) / 3600.0 AS diff_h
          FROM cams_radiation cr
         WHERE cr.location = v_location
      ) g
     WHERE diff_h > _interval_h * 1.5
     LIMIT 5;


    -- Assemble the "Findings" list + status.
    _checks := '[]'::jsonb;

    IF _neg_ghi > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'ghi', _neg_ghi))); _has_error := true; END IF;
    IF _neg_dhi > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'dhi', _neg_dhi))); _has_error := true; END IF;
    IF _neg_dni > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'dni', _neg_dni))); _has_error := true; END IF;
    IF _neg_gc  > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'ghi_clear', _neg_gc))); _has_error := true; END IF;
    IF _neg_dc  > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'dhi_clear', _neg_dc))); _has_error := true; END IF;
    IF _neg_dnc > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','error','msg', format('%s: %s negative values', 'dni_clear', _neg_dnc))); _has_error := true; END IF;

    IF _duplicates > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','warn','msg', format('%s duplicate timestamps', _duplicates))); END IF;
    IF _missing > 0 THEN _checks := _checks || jsonb_build_array(jsonb_build_object('level','warn','msg', format('%s missing intervals', _missing))); END IF;

    FOR _rec IN
        SELECT to_char(g.ts, 'YYYY-MM-DD HH24:MI:SS') AS ts, round(g.diff_h, 2) AS hours
          FROM (
            SELECT start_ts_utc AS ts,
                   EXTRACT(EPOCH FROM (start_ts_utc - LAG(start_ts_utc)
                                       OVER (ORDER BY start_ts_utc))) / 3600.0 AS diff_h
              FROM cams_radiation cr WHERE cr.location = v_location
          ) g
         WHERE g.diff_h > _interval_h * 1.5
         ORDER BY g.diff_h DESC
         LIMIT 5
    LOOP
        _checks := _checks || jsonb_build_array(
            jsonb_build_object('level','warn','msg',
                               format('gap of %s hours after %s', _rec.hours, _rec.ts)));
    END LOOP;

    IF _rel_low_pct > 5.0 THEN
        _checks := _checks || jsonb_build_array(jsonb_build_object('level','info','msg',
            format('%s%% of intervals have reliability < 1.0', _rel_low_pct)));
    END IF;

    _status := CASE WHEN _has_error THEN 'issues' ELSE 'good' END;

    RETURN jsonb_build_object(
        'span', jsonb_build_object(
            'start', to_char(_start, 'YYYY-MM-DD HH24:MI:SS'),
            'end',   to_char(_end,   'YYYY-MM-DD HH24:MI:SS'),
            'interval_h', _interval_h,
            'rows', _n),
        'time', jsonb_build_object(
            'expected_intervals', _expected,
            'rows', _n,
            'duplicates', _duplicates,
            'missing_intervals', _missing,
            'completeness_pct', _completeness,
            'gaps', _gaps,
            'timezone_utc_aware', true),
        'radiation', jsonb_build_object(
            'negatives', jsonb_build_object(
                'ghi', _neg_ghi, 'dhi', _neg_dhi, 'dni', _neg_dni,
                'ghi_clear', _neg_gc, 'dhi_clear', _neg_dc, 'dni_clear', _neg_dnc),
            'ranges', jsonb_build_object(
                'ghi', jsonb_build_array(_min_ghi, _max_ghi),
                'dhi', jsonb_build_array(_min_dhi, _max_dhi),
                'dni', jsonb_build_array(_min_dni, _max_dni),
                'ghi_clear', jsonb_build_array(_min_gc, _max_gc),
                'dhi_clear', jsonb_build_array(_min_dc, _max_dc),
                'dni_clear', jsonb_build_array(_min_dnc, _max_dnc)),
            'ghi_conservation', jsonb_build_object('mean_residual', NULL, 'max_abs_residual', NULL),
            'dhi_le_ghi_violations', _dhi_le_ghi,
            'bhi_le_ghi_violations', NULL),
        'reliability', jsonb_build_object(
            'min', _rel_min, 'median', _rel_median,
            'below_1', _rel_below1, 'below_0_5', _rel_below05,
            'low_pct', _rel_low_pct),
        'checks', _checks,
        'status', _status
    );
END;
$$;

