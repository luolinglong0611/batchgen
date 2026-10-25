"""Validate the owned decode-GC phase lifecycle, including teardown after BCT."""
import math
import statistics


def analyze(log, start, end, expected_steps, expected_ranks=16):
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError('Positive finite measurement window required')
    by_rank = {rank: [] for rank in range(expected_ranks)}
    for line in log.splitlines():
        if '[R1_DECODE_GC] ' not in line:
            continue
        import json
        row = json.loads(line.split('[R1_DECODE_GC] ', 1)[1])
        rank = row['rank']
        if (row['schema'] != 'r1-decode-gc-scope-v1' or type(rank) is not int
                or rank not in by_rank or not math.isfinite(row['unix'])):
            raise ValueError('Invalid GC observation rank/schema/timestamp')
        by_rank[rank].append(row)
    result = {}
    for rank, rows in by_rank.items():
        rows.sort(key=lambda r: r['unix'])
        starts = [r for r in rows if r['event'] == 'activated' and start <= r['unix'] <= end]
        if len(starts) != 1:
            raise ValueError('Exactly one active owned GC scope required per rank/window')
        activated = starts[0]
        closes = [r for r in rows if r['event'] == 'closed' and r['unix'] >= activated['unix']]
        if not closes:
            raise ValueError('Frozen decode scope has no teardown proof')
        closed = closes[0]
        if any(r['event'] == 'activated' and activated['unix'] < r['unix'] < closed['unix'] for r in rows):
            raise ValueError('Overlapping GC scope lifetimes')
        if (activated['forward_call'] != 1 or activated['frozen_objects'] <= 0
                or activated['automatic_collection_enabled'] is not True
                or closed['forward_call'] != expected_steps
                or closed['remaining_frozen_objects'] != 0
                or closed['automatic_collection_enabled'] is not True
                or closed['originally_frozen_objects'] != activated['frozen_objects']):
            raise ValueError('GC ownership, collection policy or complete-forward coverage failed')
        events = [r for r in rows if r['event'] == 'collection' and start <= r['unix'] <= closed['unix']]
        for row in events:
            if (row['generation'] != 2 or type(row['frozen']) is not bool
                    or any(not math.isfinite(row[k]) or row[k] < 0 for k in ('wall_ms', 'cpu_ms'))):
                raise ValueError('Invalid full-generation GC interval')
        generations = closed['generations']
        if len(generations) != 3 or generations[2]['collections'] != len(events):
            raise ValueError('Full-generation callbacks missing from sealed phase log')
        for row in generations:
            if any(not math.isfinite(row[k]) or row[k] < 0 for k in
                   ('collections', 'wall_ms', 'cpu_ms', 'maximum_wall_ms', 'collected')):
                raise ValueError('Invalid complete GC generation counters')
        frozen = [r for r in events if r['frozen']]
        before = [r for r in events if not r['frozen']]
        result[str(rank)] = {
            'activated': activated, 'closed': closed,
            'closed_after_primary_BCT_seconds': max(0, closed['unix'] - end),
            'pre_freeze_gen2_wall_ms': sum(r['wall_ms'] for r in before),
            'while_frozen_gen2_wall_ms': sum(r['wall_ms'] for r in frozen),
            'while_frozen_gen2_max_ms': max((r['wall_ms'] for r in frozen), default=0),
            'while_frozen_gen2_collections': len(frozen), 'gen2_events': events,
        }
    return {
        'schema': 'r1-decode-gc-lifecycle-analysis-v1', 'window': [start, end],
        'expected_forward_calls_each': expected_steps, 'per_rank': result,
        'across_rank': {key: {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
                        for key in ('pre_freeze_gen2_wall_ms', 'while_frozen_gen2_wall_ms',
                                    'while_frozen_gen2_max_ms', 'while_frozen_gen2_collections')
                        for values in [[r[key] for r in result.values()]]},
        'scope': 'One serial nonpersistent decode scope per worker. Activation must finish inside the original measurement window; its first following teardown may finish after BCT and remains explicit. Young-cycle collection stays enabled. Per-rank GC intervals overlap GPU/other ranks and must not be added or subtracted into exact BCT savings.',
    }
