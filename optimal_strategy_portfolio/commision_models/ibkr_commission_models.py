import numpy as np

IBKR_COMMISSION_STRUCTURE = (
    ('COMMISSION_PER_SHARE', 0.05),
    ('MINIMUM_COMMISSION_PER_ORDER', 1),
    ('MAXIMUM_PERCENT_COMMISSION', 0.01),
)


def ibkr_commission(trade_values, trade_share_amounts):
    struct = dict(IBKR_COMMISSION_STRUCTURE)
    cpr = struct['COMMISSION_PER_SHARE']
    min_c = struct['COMMISSION_PER_SHARE']
    max_p_c = struct['MAXIMUM_PERCENT_COMMISSION']

    comm = np.maximum(cpr * trade_share_amounts, min_c)
    comm = np.minimum(trade_values * max_p_c, comm)

    return comm
