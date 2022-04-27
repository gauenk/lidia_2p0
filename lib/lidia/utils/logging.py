
def print_extrema(name,tensor,drop_zero=False):
    # q_quants = th.FloatTensor([.1,.9]).to(tensor.device)
    # if drop_zero:
    #     tensor = tensor[th.where(tensor!=0)]
    # quants = th.quantile(tensor.ravel()[:7000],q_quants)
    # quants = [q.item() for q in quants]
    # msg = "[%s]: %2.2f %2.2f %2.2f %2.2f" % (name,tmin,tmax,quants[0],quants[1])
    tmin = tensor.min().item()
    tmax = tensor.max().item()
    tmean = tensor.mean().item()
    msg = "[%s]: %2.2f %2.2f %2.2f" % (name,tmin,tmax,tmean)
    print(msg)
