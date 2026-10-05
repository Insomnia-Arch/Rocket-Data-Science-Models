"""Shared analysis groups; original replay playlist identifiers stay intact."""
PLAYLIST_GROUPS = {
    'ranked-duels': 'duels', 'unranked-duels': 'duels',
    'ranked-doubles': 'team', 'unranked-doubles': 'team',
    'ranked-standard': 'team', 'unranked-standard': 'team',
    'ranked-solo-standard': 'team',
}
GROUP_LABELS = {'all': 'All modes', 'duels': '1v1', 'team': '2v2 + 3v3'}


def group_modes(frame):
    result=frame.loc[frame.playlist.isin(PLAYLIST_GROUPS)].copy()
    result['analysis_group']=result.playlist.map(PLAYLIST_GROUPS)
    return result


def composition(frame):
    return [{'playlist':playlist,'matches':int(rows.replay_id.nunique()),
             'appearances':len(rows),'players':int(rows.player_id.nunique())}
            for playlist,rows in frame.groupby('playlist')]


def analysis_cohorts(frame):
    """Overlapping views of the same rows, never duplicated within a view."""
    yield 'all',frame
    yield from frame.groupby('analysis_group')
