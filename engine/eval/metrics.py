"""DER 指标（collar 口径与 pyannote 一致），alimeeting.py 用。"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def diarization_metrics(reference,hypothesis,duration,collar=0):
    """段格式均为 (开始秒, 结束秒, 匿名说话人)，不使用真实身份名称。"""
    refids=sorted(set(str(s[2]) for s in reference))
    hypids=sorted(set(str(s[2]) for s in hypothesis))
    # 每个标注说话人的起止边界周围排除 collar 秒，与 pyannote 指标一致。
    bounds=[float(v) for seg in reference for v in seg[:2]]
    exclusions=[(max(0,v-collar/2),min(duration,v+collar/2)) for v in bounds] if collar else []
    points={0.,float(duration)}
    for seg in reference+hypothesis:
        points.update([max(0.,min(duration,float(seg[0]))),max(0.,min(duration,float(seg[1])))])
    for s,e in exclusions:
        points.update([s,e])
    points=sorted(points)
    cells=[]
    for s,e in zip(points,points[1:]):
        if e<=s:
            continue
        mid=(s+e)/2
        if any(a<=mid<b for a,b in exclusions):
            continue
        r={str(v[2]) for v in reference if v[0]<=mid<v[1]}
        h={str(v[2]) for v in hypothesis if v[0]<=mid<v[1]}
        cells.append((e-s,r,h))
    weights=np.zeros((len(refids),len(hypids)))
    for dt,r,h in cells:
        for ri,rv in enumerate(refids):
            if rv in r:
                for hi,hv in enumerate(hypids):
                    if hv in h:
                        weights[ri,hi]+=dt
    ri,hi=linear_sum_assignment(-weights)
    mapping={hypids[h]:refids[r] for r,h in zip(ri,hi)}
    total=miss=fa=conf=0.
    ov_total=ov_miss=ov_conf=ov_fa=0.
    ov_seconds=ov_detected=ov_predicted=ov_false=0.
    for dt,r,h in cells:
        nref,nhyp=len(r),len(h)
        correct=sum(mapping.get(v) in r for v in h)
        m=max(0,nref-nhyp)*dt
        f=max(0,nhyp-nref)*dt
        c=(min(nref,nhyp)-correct)*dt
        total+=nref*dt
        miss+=m
        fa+=f
        conf+=c
        if nhyp>=2:
            ov_predicted+=dt
        if nref>=2:
            ov_total+=nref*dt
            ov_seconds+=dt
            ov_miss+=m
            ov_conf+=c
            ov_fa+=f
            if nhyp>=2:
                ov_detected+=dt
        elif nhyp>=2:
            ov_false+=dt
    return dict(collar=collar,reference_speaker_seconds=total,der=(miss+fa+conf)/total if total else None,
        missed_speaker_seconds=miss,false_alarm_seconds=fa,confused_speaker_seconds=conf,
        overlap_seconds=ov_seconds,overlap_reference_speaker_seconds=ov_total,
        overlap_miss_rate=ov_miss/ov_total if ov_total else None,overlap_confusion_rate=ov_conf/ov_total if ov_total else None,
        overlap_der=(ov_miss+ov_fa+ov_conf)/ov_total if ov_total else None,
        overlap_recall=ov_detected/ov_seconds if ov_seconds else None,
        overlap_precision=(ov_predicted-ov_false)/ov_predicted if ov_predicted else 0.)

