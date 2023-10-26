from pydantic import BaseModel


class LoanPrediction(BaseModel):
    amount_requested: float
    risk_score: float
    debt_to_income_ratio: float
    employment_length: float
    year: float
    risk_times_dti: float


class GradePrediction(BaseModel):
    home_ownership: str
    initial_list_status: str
    verification_status_joint: str
    disbursement_method: str
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: float
    installment: float
    emp_length: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    mths_since_last_delinq: float
    mths_since_last_record: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    out_prncp: float
    out_prncp_inv: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    last_pymnt_amnt: float
    mths_since_last_major_derog: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_bc_dlq: float
    mths_since_recent_inq: float
    mths_since_recent_revol_delinq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    Year_issue_d: float
    cos_Month_issue_d: float
    sin_Month_issue_d: float
    log_loan_amnt: float
    risk_score: float


class SubGradePrediction(BaseModel):
    home_ownership: str
    initial_list_status: str
    verification_status_joint: str
    disbursement_method: str
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: float
    installment: float
    emp_length: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    mths_since_last_delinq: float
    mths_since_last_record: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    out_prncp: float
    out_prncp_inv: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    last_pymnt_amnt: float
    mths_since_last_major_derog: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_bc_dlq: float
    mths_since_recent_inq: float
    mths_since_recent_revol_delinq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    Year_issue_d: float
    cos_Month_issue_d: float
    sin_Month_issue_d: float
    log_loan_amnt: float
    risk_score: float


class InterestRatePrediction(BaseModel):
    home_ownership: str
    verification_status_joint: str
    delinq_2yrs: float
    mths_since_last_record: float
    open_acc: float
    pub_rec: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    mths_since_last_major_derog: float
    tot_coll_amt: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    open_rv_12m: float
    inq_fi: float
    total_cu_tl: float
    mort_acc: float
    mths_since_recent_bc_dlq: float
    mths_since_recent_revol_delinq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_op_rev_tl: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_90g_dpd_24m: float
    pub_rec_bankruptcies: float


verification_status_joint_dict = {
    "Not Verified": "Not Verified",
    "Verified": "Verified",
    "Source Verified": "Source Verified",
}


home_ownership_dict = {
    "MORTGAGE": "MORTGAGE",
    "OWN": "OWN",
    "RENT": "RENT",
    "ANY": "ANY",
    "NONE": "NONE",
    "OTHER": "OTHER",
}

initial_list_status_dict = {
    "w": "w",
    "f": "f",
}

disbursement_method_dict = {
    "Cash": "Cash",
    "DirectPay": "DirectPay",
}
