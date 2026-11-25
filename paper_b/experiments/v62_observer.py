"""Read-only V6.2 observer: immediate-source accounting, not provenance."""
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import patch
import math
import numpy as np
import model.InfoSourceSamplingLearning as m
from paper_b.experiments.v62_policies import uniform_probabilities

CHECKPOINTS = (0,1,2,3,6,11,26,51,100,200,400)
WINDOWS = ("early","middle","late")
COMPONENTS = ("prior","expert","jammer","peer")
WINDOW_FIELDS = ("opportunities","expected_requests","requests","delivered","Q","B",
                 "Lnum","snapshot_B","snapshot_Lnum","snapshot_Q","observations",
                 "gain","signed_update","raw_draw_Lnum","prior_precision")

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def close(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(a-b)<=1e-9+1e-10*max(1,abs(a),abs(b))

def precision(sd):
    return 1.0/max(float(sd)**2,m.MIN_VAR)

def account_step(parts, phi, increments, raw_phi, stored_phi):
    require(close(sum(parts.values())+sum(increments.values()),
                  sum(parts.get(k,0)+increments.get(k,0) for k in COMPONENTS)), "Accounting keys")
    require(phi>0 and raw_phi>=phi and stored_phi>0, "Invalid precision")
    scale=stored_phi/raw_phi
    return {k:(parts.get(k,0)+increments.get(k,0))*scale for k in COMPONENTS}

def window(t):
    return "early" if t<=10 else "middle" if t<=100 else "late"

def jammer_rank(ranking):
    return next(i+1 for i,s in enumerate(ranking) if s.type_of_agent=="disruptivejammer")

def source_class(source, citizen, gateways):
    if source.type_of_agent=="infoprovider":
        return "expert","expert"
    if source.type_of_agent=="disruptivejammer":
        return "jammer","jammer"
    require(source.type_of_agent=="citizen","Unexpected source")
    if gateways is not None:
        return "peer","gateway_peer" if int(source.pos) in gateways else "non_gateway_peer"
    return "peer","same_peer" if source.group_id==citizen.group_id else "other_peer"

def drift_condition(pre, post, peer_move, jammer_q):
    return jammer_q==0 and (4-pre)*peer_move>0 and abs(post-4)<abs(pre-4)

class Observer:
    def __init__(self, variant, gateways, history_writer=None):
        self.variant=variant
        self.gateways=gateways
        self.writer=history_writer
        self.states={}
        self.windows=defaultdict(lambda:[0.0]*len(WINDOW_FIELDS))
        self.audit=[]
        self.checkpoints=[]
        self.floor_events=0
        self.max_state_residual=0.0
        self.max_precision_residual=0.0
        self.periods=0

    def checkpoint(self, h, pos, mu, sd, rank, latent_rank):
        self.checkpoints.append(dict(h=h,citizen_pos=pos,mu=mu,sd=sd,
                                    rank=rank,latent_rank=latent_rank))

    def record(self, c):
        t=int(c.model.period)
        pos=int(c.pos)
        pre=float(c.mu_theta); sd=float(c.sd_theta); phi=precision(sd)
        post=float(c._pending_mu_theta); post_sd=float(c._pending_sd_theta)
        stored=precision(post_sd)
        audit=c.theta_or_delta==1
        pre_rank=jammer_rank(c._behavioral_ranking())
        latent_pre=jammer_rank(c.credibility_ranked_sources)
        pending=c._pending_ranking
        latent_post=jammer_rank(pending) if pending is not None else latent_pre
        post_rank=latent_post if pending is not None and (c.model.reliance_mode!="frozen" or t==0) else pre_rank
        if t==0:
            require(pos not in self.states and audit,"Initial audit")
            peer_ids=[s for s in c.info_source if s.type_of_agent=="citizen"]
            self.states[pos]=dict(citizen_pos=pos,group=int(c.group_id),
                role=("gateway" if pos in self.gateways else "non_gateway") if self.gateways is not None else "",
                initial_mu=pre,initial_sd=sd,initial_phi=phi,
                initial_rank=post_rank,initial_latent_rank=latent_post,
                initial_policy=None,same_peer_count=sum(s.group_id==c.group_id for s in peer_ids),
                gateway_peer_count=sum(int(s.pos) in self.gateways for s in peer_ids) if self.gateways is not None else None,
                peer_ids=[int(s.pos) for s in peer_ids],
                B_adjusted=dict(prior=phi*pre,expert=0.,jammer=0.,peer=0.),
                B_raw=dict(prior=phi*pre,expert=0.,jammer=0.,peer=0.),
                Q=dict(expert=0.,jammer=0.,peer=0.),
                first_drift=None,first_strict_drift=None,first_direct=None,
                promotion=None,promotion_pre_mu=None,promotion_pre_sd=None,
                promotion_jammer_Q=None,promotion_drift=None,promotion_strict_drift=None,
                postpromo_opportunities=0,postpromo_first_uptake=None,
                postpromo_Q=0.,postpromo_last_mu=None,floor_events=0,
                audit_count=0,theta_count=0)
            self.checkpoint(0,pos,pre,sd,pre_rank,latent_pre)
        s=self.states[pos]
        require(sum(c.num_request)==20,"Request budget")
        require(len(c._sample_order)==len(c.info_source)==len(c.sampled_msgs),"Source slots")
        require(close(sum(c.substantive_reliance_probabilities.values()),1.),"Policy mass")
        jammer=next(x for x in c.info_source if x.type_of_agent=="disruptivejammer")
        if t==1:
            s["initial_policy"]=float(c.substantive_reliance_probabilities[jammer])
        if self.variant=="neutral":
            require(all(close(p,1/len(c.info_source)) for p in c.substantive_reliance_probabilities.values()),"Neutral mass")
            if not audit:
                require([int(x.pos) for x in c._sample_order]==sorted(int(x.pos) for x in c.info_source),"Neutral source order")
        q_by=dict(expert=0.,jammer=0.,peer=0.)
        b_by=dict(expert=0.,jammer=0.,peer=0.)
        raw_phi=phi+math.fsum(c._pending_evidence_precision.values())
        peer_move=0.
        for source,msgs,n_req in zip(c._sample_order,c.sampled_msgs,c.num_request):
            n=len(msgs)
            require(n==int(n_req),"Active live delivery mismatch")
            kind,cls=source_class(source,c,self.gateways)
            q=float(c._pending_evidence_precision.get(source,0.))
            require(q>=0 and math.isfinite(q),"Assigned precision")
            if audit or n==0:
                require(q==0,"Unexpected precision")
            else:
                variance=(float(c._delivered_peer_signal_sds[source])**2+float(c.model.tau_social)**2
                    if kind=="peer" else float(source._message_sd)**2/(1 if self.variant=="RB" else n))
                require(close(q,1/max(variance,m.MIN_VAR)),"Wrong assigned source precision")
            mean=float(np.asarray(msgs,dtype=float).mean()) if n else 0.
            if n:
                require(math.isfinite(mean),"Nonfinite message")
            q_by[kind]+=q
            b_by[kind]+=q*mean
            if kind=="peer":
                snapshot=float(source._message_mu)
                snapshot_sd=float(source._message_sd)
                require(math.isfinite(snapshot) and close(snapshot_sd,c._delivered_peer_signal_sds[source]),"Peer snapshot mismatch")
                peer_move+=q*(mean-pre)/raw_phi
            else:
                snapshot=0.
            if t==0:
                v=m._observation_mean_variance(msgs,single_message_variance=c.model.single_message_variance)[1] if n else None
                self.audit.append(dict(citizen_pos=pos,source_pos=int(source.pos),source_class=cls,
                    requested=int(n_req),delivered=n,message_mean=mean if n else None,
                    sample_mean_variance=v,snapshot_mu=snapshot if kind=="peer" else None,
                    delta_pre=float(c.mu_delta[source]),delta_sd_pre=float(c.sd_delta[source]),
                    delta_post=float(c._pending_mu_delta[source]),delta_sd_post=float(c._pending_sd_delta[source]),
                    rank_pre=c._behavioral_ranking().index(source)+1,
                    rank_post=pending.index(source)+1))
            if not audit:
                vals=self.windows[pos,window(t),cls]
                row=(1.,20*float(c.substantive_reliance_probabilities[source]),int(n_req),n,
                     q,q*mean,q*mean*mean,q*snapshot if kind=="peer" else 0.,
                     q*snapshot*snapshot if kind=="peer" else 0.,q if kind=="peer" else 0.,
                     int(n>0),q/raw_phi,q*(mean-pre)/raw_phi,
                     q*math.fsum(float(x)**2 for x in msgs)/n if n else 0.,
                     phi)
                for j,x in enumerate(row):
                    vals[j]+=x
        if audit:
            s["audit_count"]+=1
            require(close(pre,post) and close(phi,stored),"Audit changed state")
            if t>=1 and s["initial_rank"]>=3 and s["promotion"] is None and pre_rank!=1 and post_rank==1:
                s.update(promotion=t,promotion_pre_mu=pre,promotion_pre_sd=sd,
                         promotion_jammer_Q=s["Q"]["jammer"],promotion_drift=s["first_drift"],
                         promotion_strict_drift=s["first_strict_drift"])
        else:
            s["theta_count"]+=1
            prior_j=s["Q"]["jammer"]
            if q_by["jammer"]>0 and s["first_direct"] is None:
                s["first_direct"]=t
            if s["initial_rank"]>=3 and s["promotion"] is None and drift_condition(pre,post,peer_move,q_by["jammer"]):
                if s["first_drift"] is None:
                    s["first_drift"]=t
                if prior_j==0 and s["first_strict_drift"] is None:
                    s["first_strict_drift"]=t
            if s["promotion"] is not None and t>s["promotion"] and s["postpromo_opportunities"]<5:
                s["postpromo_opportunities"]+=1
                s["postpromo_Q"]+=q_by["jammer"]
                s["postpromo_last_mu"]=post
                if q_by["jammer"]>0 and s["postpromo_first_uptake"] is None:
                    s["postpromo_first_uptake"]=t
            for kind in q_by:
                s["Q"][kind]+=q_by[kind]
                s["B_raw"][kind]+=b_by[kind]
            if raw_phi>1/m.MIN_VAR:
                self.floor_events+=1
                s["floor_events"]+=1
            else:
                require(close(raw_phi,stored),"Posterior precision mismatch")
            s["B_adjusted"]=account_step(s["B_adjusted"],phi,b_by,raw_phi,stored)
        reconstructed=math.fsum(s["B_adjusted"].values())/stored
        residual=abs(reconstructed-post)
        self.max_state_residual=max(self.max_state_residual,residual)
        require(close(reconstructed,post),"Signed accounting mismatch")
        if not s["floor_events"]:
            expected_phi=s["initial_phi"]+math.fsum(s["Q"].values())
            self.max_precision_residual=max(self.max_precision_residual,abs(stored-expected_phi))
            require(close(stored,expected_phi),"Cumulative precision mismatch")
        s.update(mu=post,sd=post_sd,rank=post_rank,latent_rank=latent_post)
        if t+1 in CHECKPOINTS:
            self.checkpoint(t+1,pos,post,post_sd,post_rank,latent_post)
        if self.writer is not None:
            self.writer.writerow((pos,t,int(audit),pre,sd,post,post_sd,pre_rank,post_rank,
                latent_pre,latent_post,q_by["expert"],q_by["jammer"],q_by["peer"],
                b_by["expert"],b_by["jammer"],b_by["peer"],peer_move,
                s["Q"]["jammer"]))
        self.periods+=1

    def finish(self,out,horizon):
        require(len(self.states)==100 and self.periods==100*horizon,"Incomplete observer")
        b={int(x["citizen_pos"]):x for x in out["beliefs"]}
        q={int(x["citizen_pos"]):x for x in out["channel_citizens"]}
        terminal=[]
        for pos,s in sorted(self.states.items()):
            require(close(s["mu"],float(b[pos]["terminal_mu_theta"])) and close(s["sd"],float(b[pos]["terminal_sd_theta"])),"Terminal state mismatch")
            for cls in ("expert","jammer"):
                require(close(s["Q"][cls],float(q[pos]["Q_"+cls])),"Terminal external Q")
            require(close(s["Q"]["peer"],float(q[pos]["Q_same_peer"])+float(q[pos]["Q_other_peer"])),"Terminal peer Q")
            require(s["audit_count"]+s["theta_count"]==horizon,"Learning count")
            s["sequence"]=int(s["promotion_drift"] is not None and s["postpromo_first_uptake"] is not None)
            s["strict_sequence"]=int(s["sequence"] and s["promotion_strict_drift"] is not None and s["promotion_jammer_Q"]==0)
            s["censored"]=int(s["promotion"] is not None and s["postpromo_opportunities"]<5)
            s["W_jammer"]=float(q[pos]["W_jammer"])
            s["phi"]=precision(s["sd"])
            s["signed"]={k:v/s["phi"] for k,v in s["B_adjusted"].items()}
            s["signed_mse"]={k:s["mu"]*v for k,v in s["signed"].items()}
            require(close(sum(s["signed_mse"].values()),s["mu"]**2),"MSE account")
            terminal.append(s)
        return dict(terminal=terminal,initial_audit=self.audit,checkpoints=self.checkpoints,
            windows=[dict(citizen_pos=p,window=w,source_class=c,**dict(zip(WINDOW_FIELDS,vals)))
                     for (p,w,c),vals in sorted(self.windows.items())],
            validation=dict(periods=self.periods,floor_events=self.floor_events,
                max_state_residual=self.max_state_residual,max_precision_residual=self.max_precision_residual))

@contextmanager
def observe(observer):
    original=m.Citizen.compute_pending_update
    def wrapped(self):
        original(self)
        observer.record(self)
    with patch.object(m.Citizen,"compute_pending_update",wrapped):
        yield
