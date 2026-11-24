"""V6 prospective sensitivity: cap only substantive external evidence."""
import math
import numpy as np
import model.InfoSourceSamplingLearning as m

CAP = 5


def capped_theta_update(self):
    """Same source-posterior Gaussian update, except external n_eff=min(n,5).

    Keep the sample mean of ALL delivered draws. No message generation,
    request allocation, credibility audit, or peer precision change.
    """
    if self.model.peer_evidence_mode != "source_posterior" or self.theta_or_delta != 0:
        raise RuntimeError("Precision cap is scoped to substantive source-posterior updates")
    prior_mu = float(self.mu_theta_beliefs[-1])
    prior_var = max(float(self.sd_theta_beliefs[-1])**2,m.MIN_VAR)
    precision = 1.0/prior_var
    weighted_mean = prior_mu/prior_var
    used = 0
    for source,messages in zip(self._sample_order,self.sampled_msgs):
        values = np.asarray(list(messages),dtype=float)
        if values.size == 0:
            continue
        obs_mu = float(values.mean())
        if source.type_of_agent == "citizen":
            obs_var = max(float(self._delivered_peer_signal_sds[source])**2
                          + float(self.model.tau_social)**2,m.MIN_VAR)
        elif source.type_of_agent in ("infoprovider","disruptivejammer"):
            source_var = max(float(source._message_sd)**2,m.MIN_VAR)
            obs_var = max(source_var/min(int(values.size),CAP),m.MIN_VAR)
        else:
            raise RuntimeError("Unknown evidence source")
        added_precision = 1.0/obs_var
        precision += added_precision
        weighted_mean += obs_mu/obs_var
        self._pending_evidence_precision[source] = float(added_precision)
        used += 1
    if used == 0:
        return prior_mu,math.sqrt(prior_var)
    post_var = 1.0/precision
    post_mu = post_var*weighted_mean
    return float(post_mu),math.sqrt(max(float(post_var),m.MIN_VAR))
