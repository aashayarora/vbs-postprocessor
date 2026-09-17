import numpy as np
import torch

def distance_corr(
        var_1:torch.tensor,
        var_2:torch.tensor,
        normedweight:torch.tensor,
        power=1,
) -> torch.tensor:

    # Normalize the weights
    normedweight = normedweight/torch.sum(normedweight)*len(var_1)

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if (power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif (power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    return dCorr

def __check_number_of_events(var1, var2, rand_cuts, n_events_min):

    cut_var1, cut_var2 = rand_cuts

    NA_diff = torch.sum((torch.sigmoid(100*(var1 - cut_var1))*torch.sigmoid(100*(cut_var2 - var2))))
    NB_diff = torch.sum((torch.sigmoid(100*(var1 - cut_var1))*torch.sigmoid(100*(var2 - cut_var2))))
    NC_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var1))*torch.sigmoid(100*(cut_var2 - var2))))
    ND_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var1))*torch.sigmoid(100*(var2 - cut_var2))))

    check = (
        NA_diff > n_events_min
        and NB_diff > n_events_min
        and NC_diff > n_events_min
        and ND_diff > n_events_min
    )

    return check

def __get_abcd_random_cuts(var1, var2, n_events_min, max_tries=1_000_000_000):

    for _ in range(max_tries):
        x_min = torch.quantile(var1, 0.01).item()
        x_max = torch.quantile(var1, 0.99).item()
        y_min = torch.quantile(var2, 0.01).item()
        y_max = torch.quantile(var2, 0.99).item()
        if not (np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)):
            raise ValueError("Non-finite cut bounds")
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("Degenerate cut bounds")

        rand_cut_x = np.random.uniform(x_min, x_max)
        rand_cut_y = np.random.uniform(y_min, y_max)
        rand_cuts = [rand_cut_x, rand_cut_y]

        check = __check_number_of_events(
            var1,
            var2,
            rand_cuts,
            n_events_min,
        )

        if check:
            return rand_cuts

    raise ValueError(f"Could not find a suitable set of cuts after {max_tries} tries")

def closure(var_1, var_2, weights, labels, symmetrize, n_events_min=10):
    eps = 1e-12
    var_1 = var_1[labels == 0]
    var_2 = var_2[labels == 0]
    weights = weights[labels == 0]

    finite_mask = torch.isfinite(var_1) & torch.isfinite(var_2) & torch.isfinite(weights)
    var_1 = var_1[finite_mask]
    var_2 = var_2[finite_mask]
    weights = torch.clamp(weights[finite_mask], min=0.0)

    if len(var_1) <= max(4, n_events_min):
        return var_1.new_tensor(0.0)

    cut_var1, cut_var2 = __get_abcd_random_cuts(var_1, var_2, n_events_min)

    NA_diff = torch.sum((torch.sigmoid(100*(var_1 - cut_var1))*torch.sigmoid(100*(cut_var2 - var_2)))*weights)
    NB_diff = torch.sum((torch.sigmoid(100*(var_1 - cut_var1))*torch.sigmoid(100*(var_2 - cut_var2)))*weights)
    NC_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var_1))*torch.sigmoid(100*(cut_var2 - var_2)))*weights)
    ND_diff = torch.sum((torch.sigmoid(100*(cut_var1 - var_1))*torch.sigmoid(100*(var_2 - cut_var2)))*weights)

    if symmetrize:
        closure_loss = torch.abs(NA_diff * ND_diff - NB_diff * NC_diff) / torch.clamp(
            NA_diff * ND_diff + NB_diff * NC_diff,
            min=eps,
        )
    else:
        closure_loss = torch.abs(NA_diff * ND_diff - NB_diff * NC_diff) / torch.clamp(
            NC_diff * NB_diff,
            min=eps,
        )

    return torch.nan_to_num(closure_loss, nan=0.0, posinf=1.0, neginf=0.0)
