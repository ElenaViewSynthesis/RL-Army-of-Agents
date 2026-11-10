"""Torch-free acceptance of saved GPU evidence; process success alone is insufficient."""
import math


def review(run_id, backends, stages, results, names, environment, probe, stock_profile):
    failures, anomalies = [], []
    if not stages or any(s.get('returncode') != 0 or s.get('timed_out') for s in stages):
        failures.append('one or more stages failed or timed out')
    runtime = {k:v for k,v in probe.items() if k.startswith('runtime/')}
    matches = bool(runtime) and all(environment.get(k) == v for k,v in runtime.items())
    if not matches:
        failures.append('packaged environment differs from fresh require probe')
    if environment.get('run_id') != run_id:
        failures.append('packaged environment has wrong run_id')
    rollouts = results.get('rollouts', {})
    profiles = results.get('profiles', {})
    def timing(case, row, prefix=''):
        for field in ('t_eager_us','t_compile_us','t_gen_us'):
            value = row.get(prefix+field)
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value <= 0:
                anomalies.append({'case':case,'key':prefix+field,'value':value})
    def capture(case, profile):
        timing(case, profile)
        if profile.get('timing_measured') is not True or profile.get('profile/diagnostic'):
            failures.append(f'{case}: no valid reward profile')
        for session in ('eager','compile','gen'):
            count = profile.get(f'profile/{session}_events')
            if type(count) is not int or count <= 0:
                failures.append(f'{case}: missing/empty {session} event count')
    for case, row in rollouts.items():
        if case in ('timeout','max_turns'):
            if row.get('reward/reason') != case or row.get('reward/value') != -1:
                failures.append(f'{case}: expected rejection missing')
            continue
        timing(case,row,'env/')
        profile = profiles.get(case,{})
        capture(case,profile)
        hack = case == 'hack' or case.startswith(('hack-','reward-hack-torchop'))
        expected = 0.0 if hack else 1.0
        if row.get('env/verify_ok') is not True or row.get('env/provenance') != expected:
            failures.append(f'{case}: expected verified provenance {expected}')
        if hack and (row.get('reward/value') != -1 or row.get('reward/reason') != 'provenance'):
            failures.append(f'{case}: hack rejection missing')
        if not hack and (row.get('reward/base',-1) <= 0 or row.get('reward/reason') in ('timing','unmeasured')):
            failures.append(f'{case}: honest reward invalid')
    for backend in backends:
        if backend not in rollouts:
            failures.append(f'{backend}: honest case absent')
        hack = 'reward-hack-torchop' if backend == 'triton' else f'reward-hack-torchop-{backend}'
        if hack not in rollouts:
            failures.append(f'{backend}: hack case absent')
        row = names.get('backends',{}).get(backend,{})
        if row.get('measured') is not True or row.get('provenance') != 1.0 or row.get('unmatched_agent_names') != []:
            failures.append(f'{backend}: kernel-name evidence incomplete')
        capture('kernel_names/'+backend,row.get('profile',{}))
    if 'cuda' in backends:
        capture('stock_pipeline',stock_profile)
    if anomalies:
        failures.append('nonpositive, missing or nonfinite timing')
    return {'run_id':run_id, 'status':'partial' if failures else 'pass',
            'scope':list(backends), 'failures':failures, 'timing_anomalies':anomalies,
            'environment_matches_require_probe':matches, 'stage_statuses':stages,
            'rollouts':rollouts}
