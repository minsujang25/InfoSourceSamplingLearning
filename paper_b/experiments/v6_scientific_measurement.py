"""Passive post-run summaries for v6. No simulation state or RNG writes."""
from collections import defaultdict
import math
import statistics

CLASSES = ("expert", "same_peer", "other_peer", "jammer")
ALL_CLASSES = (*CLASSES, "gateway_peer")
TARGETS = ("all", "credibility", "theta")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def groups_for(citizens, gateways):
    groups = defaultdict(list)
    for c in citizens:
        pos = int(c.pos)
        groups[("all", "all")].append(pos)
        groups[("citizen_group", str(c.group_id))].append(pos)
        if gateways is not None:
            groups[("gateway_role", "gateway" if pos in gateways else "non_gateway")].append(pos)
    return dict(groups)


def belief_rows(citizens, groups, steps, truth):
    by_pos = {int(c.pos): c for c in citizens}
    out = []
    for step in steps:
        for (kind, group), ids in sorted(groups.items()):
            mu = [float(by_pos[i].mu_theta_beliefs[step]) for i in ids]
            sd = [float(by_pos[i].sd_theta_beliefs[step]) for i in ids]
            require(all(math.isfinite(v) for v in mu+sd) and min(sd)>0, "Invalid belief history")
            mean = statistics.mean(mu)
            var = statistics.pvariance(mu)
            mse = statistics.mean((v-truth)**2 for v in mu)
            bias2 = (mean-truth)**2
            require(math.isclose(mse, bias2+var, rel_tol=1e-10, abs_tol=1e-10), "MSE identity")
            out.append(dict(horizon_step=step, period=step-1, group_type=kind, group=group,
                            n_citizens=len(ids), mean_belief=mean, squared_bias=bias2,
                            belief_variance=var, mse_truth=mse, mean_posterior_sd=statistics.mean(sd)))
    return out


def classify(edge, gateways):
    t = edge["source_type"]
    if t == "infoprovider":
        return "expert"
    if t == "disruptivejammer":
        return "jammer"
    require(t == "citizen", "Unknown structural source class")
    return "same_peer" if edge["ego_group"] == edge["source_group"] else "other_peer"


def request_rows(history, groups, steps, credit, gateways=None):
    """Reconstruct integer allocations from X; no delivered-message claim.

    X is counts/C by construction in Citizen.sample_messages(). Track the
    full history, split audits and substantive periods, and save cumulative
    snapshots. Gateway peer count is a subset, not a fifth disjoint class.
    """
    memberships = defaultdict(list)
    for key, ids in groups.items():
        for ego in ids:
            memberships[ego].append(key)
    require(len(memberships)==len(groups[("all","all")]), "Group membership mismatch")
    keys = [(kind, group, target) for kind, group in sorted(groups) for target in TARGETS]
    totals = {key: [0]*(len(ALL_CLASSES)+1) for key in keys}
    cls_index = {c:i for i,c in enumerate(ALL_CLASSES)}
    rows = []
    wanted = set(steps)

    def snapshot(step):
        for kind, group, target in keys:
            counts = totals[(kind,group,target)]
            total = sum(counts[:4])
            n = len(groups[(kind,group)])
            require(total==counts[-1]*credit, "Request budget identity")
            row = dict(horizon_step=step, period=step-1, group_type=kind, group=group,
                       learning_target=target, n_citizens=n, citizen_periods=counts[-1],
                       total_requests=total)
            for cls,i in cls_index.items():
                row[f"requests_{cls}"] = counts[i]
                row[f"request_share_{cls}"] = counts[i]/total if total else None
            rows.append(row)
        for kind, group in groups:
            a,b,c = (totals[(kind,group,t)] for t in TARGETS)
            require(all(x==y+z for x,y,z in zip(a,b,c)), "Audit/theta accounting mismatch")
            require(a[-1]==len(groups[(kind,group)])*step, "Citizen-period count mismatch")

    if 0 in wanted:
        snapshot(0)
    for period, edges in enumerate(history):
        by_ego = {}
        targets = {}
        for e in edges:
            ego = int(e["ego"])
            require(int(e["period"])==period, "Reliance history period mismatch")
            target = str(e["learning_target"])
            require(target in TARGETS[1:], "Unknown learning target")
            if ego not in by_ego:
                by_ego[ego] = [0]*len(ALL_CLASSES)
                targets[ego] = target
            require(targets[ego]==target, "Mixed citizen learning target")
            mass = float(e["realized_reliance"])
            count = round(mass*credit)
            require(0<=count<=credit and math.isclose(mass*credit,count,abs_tol=1e-9,rel_tol=0),
                    "Noninteger request allocation")
            cls = classify(e, gateways)
            by_ego[ego][cls_index[cls]] += count
            if gateways is not None and cls in ("same_peer","other_peer") and int(e["source"]) in gateways:
                by_ego[ego][cls_index["gateway_peer"]] += count
        require(set(by_ego)==set(memberships), "Missing citizen request history")
        for ego, counts in by_ego.items():
            require(sum(counts[:4])==credit, "Citizen request budget mismatch")
            for kind, group in memberships[ego]:
                for target in ("all", targets[ego]):
                    total = totals[(kind,group,target)]
                    for i,value in enumerate(counts):
                        total[i] += value
                    total[-1] += 1
        if period+1 in wanted:
            snapshot(period+1)
    require(wanted <= set(range(len(history)+1)), "Missing request checkpoints")
    return rows


def summarize_model(model, gateways, steps, credit):
    citizens = list(model.citizens)
    horizon = int(model.max_steps)
    require(len(model.reliance_history)==horizon, "Missing reliance history")
    require(all(len(c.mu_theta_beliefs)==len(c.sd_theta_beliefs)==horizon+1 for c in citizens),
            "Missing citizen posterior histories")
    groups = groups_for(citizens, gateways)
    return (belief_rows(citizens, groups, steps, float(model.state_of_the_world)),
            request_rows(model.reliance_history, groups, steps, credit, gateways))
