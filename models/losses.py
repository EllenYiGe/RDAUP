import torch
import torch.nn.functional as F

def classification_loss(logits, labels):
    """
    Source domain classification loss (cross-entropy)
    """
    return F.cross_entropy(logits, labels)

def entropy_weight(logits):
    """
    Calculate entropy weight: w = 1 + exp(-H(x))
    Where H(x) = - sum(p_i * log p_i)
    logits: (B, C)
    """
    softmax_out = F.softmax(logits, dim=1)
    log_softmax_out = F.log_softmax(logits, dim=1)
    entropy = - (softmax_out * log_softmax_out).sum(dim=1)  # (B)
    weight = 1.0 + torch.exp(-entropy)
    return weight.detach()

def adversarial_loss_rdaup(features_s, features_t, logits_s, logits_t, domain_discriminator):
    """
    RDAUP Adversarial Loss:
    - Includes 3 terms:
      1) Source domain classified as source domain (label=1)
      2) Target domain classified as target domain (label=0)
      3) Source domain transferred to target domain (label=0)
    """
    w_s = entropy_weight(logits_s)
    w_t = entropy_weight(logits_t)

    pred_s = domain_discriminator(features_s)  # (B,1)
    pred_t = domain_discriminator(features_t)

    label_s_1 = torch.ones_like(pred_s)
    label_s_0 = torch.zeros_like(pred_s)
    label_t_0 = torch.zeros_like(pred_t)

    # Source domain -> Source domain
    loss_s_as_source = F.binary_cross_entropy_with_logits(
        pred_s, label_s_1, weight=w_s.unsqueeze(1))
    # Target domain -> Target domain
    loss_t_as_target = F.binary_cross_entropy_with_logits(
        pred_t, label_t_0, weight=w_t.unsqueeze(1))
    # Source domain -> Target domain (transfer)
    loss_s_as_target = F.binary_cross_entropy_with_logits(
        pred_s, label_s_0, weight=w_s.unsqueeze(1))

    loss_adv = loss_s_as_source + loss_t_as_target + loss_s_as_target
    return loss_adv

def entropy_minimization_loss(logits_t):
    """
    Target domain entropy minimization: E_{x_t}[ H(softmax(logits_t)) ]
    """
    softmax_out = F.softmax(logits_t, dim=1)
    log_softmax_out = F.log_softmax(logits_t, dim=1)
    ent = - (softmax_out * log_softmax_out).sum(dim=1).mean()
    return ent

def uncertainty_penalization_loss(logits, labels):
    """
    Uncertainty Penalization (Equation (7) from the paper):
    - Apply additional penalty for incorrect classes j != g:
      sum_j( p_j/(1-p_g)* log( p_j/(1-p_g) ) ), take negative and average
    """
    p = F.softmax(logits, dim=1)
    batch_size, class_num = p.size()
    loss_upl = 0.0

    for i in range(batch_size):
        g = labels[i].item()
        p_g = p[i, g]
        sum_j = 0.0
        for j in range(class_num):
            if j == g:
                continue
            p_j = p[i, j]
            denom = 1.0 - p_g
            if denom > 1e-7:
                ratio = p_j / denom
                if ratio > 1e-7:
                    sum_j += ratio * torch.log(ratio)
        loss_upl += sum_j
    loss_upl = - loss_upl / batch_size
    return loss_upl
